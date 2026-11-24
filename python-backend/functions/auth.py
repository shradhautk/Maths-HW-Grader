"""POST /auth/* — Lambda gradeant-auth (handler auth.lambda_handler).

signup, login, logout, forgot-password, change-password, me, profile-photo.
Errors are {"message": ...}. Same behaviour as backend/src/services/auth.service.js.
Login tokens are HS256 JWTs signed with the shared key from gradeant_common.secrets
(Secrets Manager in production); there is no built-in fallback key.
"""
import json
import boto3
from botocore.config import Config
import uuid
import hashlib
import hmac
import base64
import os
import jwt
import time
from datetime import datetime

from gradeant_common import config, dynamo, secrets
from gradeant_common.http import handle
from gradeant_common.permissions import can_sign_up, role_label, self_signup_roles

table = dynamo.table('users')
TOKEN_EXPIRY = config.env_int('TOKEN_EXPIRY_SECONDS', 86400)  # 24 hours by default

# Password-reset requests are worked by the admin console (functions/admin.py), so
# both functions point at the same table (RESET_REQUESTS_TABLE).
reset_table = dynamo.table('reset_requests')

# Profile photos: private S3 bucket, browser uploads with a presigned PUT.
PROFILE_PHOTO_BUCKET = os.environ.get('PROFILE_PHOTO_BUCKET', '')
PHOTO_URL_TTL = 3600
PHOTO_TYPES = {'image/jpeg': 'jpg', 'image/png': 'png', 'image/webp': 'webp'}
PHOTO_MAX_BYTES = 5 * 1024 * 1024

def respond(status, payload):
    # CORS headers are added to every response by gradeant_common.http.handle().
    return {'statusCode': status, 'headers': {'Content-Type': 'application/json'}, 'body': json.dumps(payload)}

def password_problem(password):
    """Same rules the change-password page shows: 8+ chars, an uppercase letter, a number."""
    if len(password) < 8:
        return 'Password must be at least 8 characters.'
    if not any(c.isupper() for c in password):
        return 'Password must include an uppercase letter.'
    if not any(c.isdigit() for c in password):
        return 'Password must include a number.'
    return None

def authenticated_user(event):
    """Returns the user item for the Bearer token, or None."""
    headers = {k.lower(): v for k, v in (event.get('headers') or {}).items()}
    auth_header = headers.get('authorization', '')
    if not auth_header.startswith('Bearer '):
        return None
    payload = verify_token(auth_header.split(' ', 1)[1])
    if not payload:
        return None
    item = table.get_item(Key={'email': payload.get('email')}).get('Item')
    if not item or item.get('user_id') != payload.get('user_id'):
        return None
    if item.get('status') == 'suspended':
        return None
    return item

def _s3():
    # Signature V4: the only presigned-URL form that signs the upload size (ContentLength).
    return boto3.client('s3', region_name=config.data_region(), config=Config(signature_version='s3v4'))

def photo_view_url(user):
    key = user.get('photo_key')
    if not key or not PROFILE_PHOTO_BUCKET:
        return None
    return _s3().generate_presigned_url(
        'get_object', Params={'Bucket': PROFILE_PHOTO_BUCKET, 'Key': key}, ExpiresIn=PHOTO_URL_TTL)

def hash_password(password, salt=None):
    """Hash a password for storing."""
    print(f"Hashing password (salt provided: {salt is not None})")
    if salt is None:
        salt = os.urandom(32)  # 32 bytes = 256 bits
        print("Generated new salt for password hashing")
    
    pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
    print("Password successfully hashed")
    return salt + pwdhash

def verify_password(stored_password, provided_password):
    """Verify a stored password against a provided password"""
    print("Verifying password")
    salt = stored_password[:32]  # 32 is the length of the salt
    stored_hash = stored_password[32:]
    pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
    result = hmac.compare_digest(pwdhash, stored_hash)
    print(f"Password verification result: {result}")
    return result

def generate_token(user_id, email):
    """Generate JWT token for authenticated users"""
    print(f"Generating JWT token for user: {email} (ID: {user_id})")
    payload = {
        'user_id': user_id,
        'email': email,
        'exp': int(time.time()) + TOKEN_EXPIRY,
        'iat': int(time.time())
    }
    key = secrets.jwt_secret()
    if not key:
        raise RuntimeError('JWT_SECRET is not configured (set JWT_SECRET_ARN)')
    token = jwt.encode(payload, key, algorithm='HS256')
    print("JWT token generated successfully")
    return token

def verify_token(token):
    """Verify JWT token"""
    print("Verifying JWT token")
    try:
        key = secrets.jwt_secret()
        if not key:
            return None
        payload = jwt.decode(token, key, algorithms=['HS256'])
        print(f"Token verified successfully for user: {payload.get('email')}")
        return payload
    except jwt.ExpiredSignatureError:
        print("Token verification failed: Token expired")
        return None
    except jwt.InvalidTokenError:
        print("Token verification failed: Invalid token")
        return None

def signup(event):
    """Handle user signup"""
    print("Processing signup request")
    try:
        # (Never log the raw event: its body holds the plaintext password.)
        # Parse request body
        body = json.loads(event['body'])
        email = body['email']
        password = body['password']
        name = body.get('name', '')
        userRole=body.get('role') or 'student'
        # Who may self-sign-up comes from config/app.yaml roles.<role>.signup (admin never can).
        if not can_sign_up(userRole, 'self'):
            if can_sign_up(userRole, 'invite'):
                return respond(403, {'message': f'{role_label(userRole)} accounts are invite-only. Ask your school administrator for an invite.'})
            return respond(400, {'message': f"Role must be one of: {', '.join(self_signup_roles())}."})
        problem = password_problem(str(password))
        if problem:
            return {
                'statusCode': 400,
                'body': json.dumps({'message': problem})
            }
        print(f"Signup request for email: {email}, name: {name}")
        
        # Check if user already exists
        print(f"Checking if user {email} already exists")
        response = table.get_item(
            Key={
                'email': email
            }
        )
        
        if 'Item' in response:
            print(f"User {email} already exists - signup failed")
            return {
                'statusCode': 409,
                'body': json.dumps({'message': 'User already exists'})
            }
        
        # Hash password
        print("Hashing password for new user")
        password_hash = hash_password(password)
        
        # Create user record
        user_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        print(f"Creating new user record with ID: {user_id}")
        user_item = {
            'user_id': user_id,
            'email': email,
            'name': name,
            'password': base64.b64encode(password_hash).decode('utf-8'),
            'userRole':userRole,
            'created_at': timestamp,
            'updated_at': timestamp
        }
        
        # Save to DynamoDB
        print(f"Saving user {email} to DynamoDB")
        table.put_item(Item=user_item)
        print(f"User {email} saved successfully")
        
        # Generate token
        token = generate_token(user_id, email)
        
        print(f"Signup successful for user: {email}")
        return {
            'statusCode': 201,
            "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
                       },
            'body': json.dumps({
                'message': 'User created successfully',
                'user_id': user_id,
                'userRole':userRole,
                'name': name,
                "email": email,
                'token': token
            })
        }
    
    except KeyError as e:
        print(f"Signup error - Missing required field: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps({'message': f'Missing required field: {str(e)}'})
        }
    except Exception as e:
        print(f"Signup error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Error creating user.'})
        }

def login(event):
    """Handle user login"""
    print("Processing login request")
    try:
        # Parse request body
        body = json.loads(event['body'])
        email = body['email']
        password = body['password']
        userRole=body['userType']
        print(f"Login attempt for user: {email}")
        
        # Get user from DynamoDB
        print(f"Retrieving user {email} from DynamoDB")
        response = table.get_item(
            Key={
                'email': email
            }
        )
        
        if 'Item' not in response:
            print(f"Login failed - User {email} not found")
            return {
                'statusCode': 401,
                'body': json.dumps({'message': 'Invalid credentials'})
            }
        
        user = response['Item']
        print(f"User {email} found in database")
        
        # Verify Role
        print("Verifying Role")
        Role = user['userRole']
        if userRole!=Role:
            print(f"Login failed - Invalid Role for user {email}")
            return {
                'statusCode': 401,
                'body': json.dumps({'message': 'Invalid credentials'})
            }
            print(f"Role verified for user {email}")
        # Verify password
        print("Verifying password")
        stored_password = base64.b64decode(user['password'])
        if not verify_password(stored_password, password):
            print(f"Login failed - Invalid password for user {email}")
            return {
                'statusCode': 401,
                'body': json.dumps({'message': 'Invalid credentials'})
            }

        # An admin suspended (or deleted) this account in the admin console.
        if user.get('status') == 'suspended':
            print(f"Login refused - account {email} is suspended")
            return {
                'statusCode': 403,
                'body': json.dumps({'message': 'This account is suspended. Contact your administrator.'})
            }

        # Generate token
        print("Password verified, generating token")
        token = generate_token(user['user_id'], user['email'])
        
        # Update last login timestamp
        print(f"Updating last login timestamp for user {email}")
        table.update_item(
            Key={
                'email': email
            },
            UpdateExpression="set last_login = :login_time",
            ExpressionAttributeValues={
                ':login_time': datetime.now().isoformat()
            }
        )
        
        print(f"Login successful for user: {email}")
        return {
            'statusCode': 200,
                        "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
                       },
            'body': json.dumps({
                'message': 'Login successful',
                'user_id': user['user_id'],
                'userRole': user['userRole'],
                "name": user['name'],
                "email": user['email'],
                'must_change_password': bool(user.get('must_change_password', False)),
                'photoUrl': photo_view_url(user),
                'token': token
            })
        }
    
    except KeyError as e:
        print(f"Login error - Missing required field: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps({'message': f'Missing required field: {str(e)}'})
        }
    except Exception as e:
        print(f"Login error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Error during login.'})
        }

# def logout(event):
#     """Handle user logout"""
#     print("Processing logout request")
#     try:
#         # Parse authorization header
#         auth_header = event['headers'].get('Authorization', '')
#         print(f"Authorization header: {auth_header[:20]}...")
        
#         if not auth_header.startswith('Bearer '):
#             print("Logout failed - Invalid authorization header format")
#             return {
#                 'statusCode': 401,
#                 'body': json.dumps({'message': 'Invalid authorization header'})
#             }
        
#         token = auth_header.split(' ')[1]
#         print("Extracted token from authorization header")
        
#         payload = verify_token(token)
        
#         if not payload:
#             print("Logout failed - Invalid or expired token")
#             return {
#                 'statusCode': 401,
#                 'body': json.dumps({'message': 'Invalid or expired token'})
#             }
        
#         print(f"Token validated for user: {payload.get('email')}")
#         # For a more complete implementation, we would add the token to a blacklist in DynamoDB
#         # This would require another table to store invalidated tokens until they expire
        
#         print("Logout successful")
#         return {
#             'statusCode': 200,
#             'body': json.dumps({'message': 'Logout successful'})
#         }
    
#     except Exception as e:
#         print(f"Logout error: {str(e)}")
#         return {
#             'statusCode': 500,
#             'body': json.dumps({'message': f'Error during logout: {str(e)}'})
#         }

def logout(event):
    """Handle user logout"""
    print("Processing logout request")
    try:
        # Check if Authorization header exists
        if 'headers' not in event or not event['headers']:
            print("Logout failed - No headers in request")
            return {
                'statusCode': 401,
                'body': json.dumps({'message': 'No headers provided'})
            }
            
        auth_header = event['headers'].get('Authorization', '')
        print(f"Authorization header present: {bool(auth_header)}")
        print(f"Authorization header value: {auth_header[:20]}..." if auth_header else "None")
        
        if not auth_header or not auth_header.startswith('Bearer '):
            print("Logout failed - Invalid authorization header format")
            return {
                'statusCode': 401,
                'body': json.dumps({'message': 'Invalid authorization header'})
            }
        
        try:
            token = auth_header.split(' ')[1]
            print(f"Token extracted: {token[:10]}...")
        except IndexError:
            print("Failed to extract token from authorization header")
            return {
                'statusCode': 401,
                'body': json.dumps({'message': 'Malformed authorization header'})
            }
        
        # Check if body exists and is valid JSON (if present)
        if 'body' in event and event['body']:
            try:
                body = json.loads(event['body']) if event['body'] else {}
                print(f"Request body parsed successfully: {body}")
            except json.JSONDecodeError as e:
                print(f"Invalid JSON in request body: {str(e)}")
                return {
                    'statusCode': 400,
                    'body': json.dumps({'message': 'Invalid JSON in request body'})
                }
        
        try:
            payload = verify_token(token)
        except Exception as e:
            print(f"Token verification threw exception: {str(e)}")
            return {
                'statusCode': 401,
                'body': json.dumps({'message': 'Token verification failed.'})
            }
        
        if not payload:
            print("Logout failed - Invalid or expired token")
            return {
                'statusCode': 401,
                'body': json.dumps({'message': 'Invalid or expired token'})
            }
        
        print(f"Token validated for user: {payload.get('email')}")
        
        return {
            'statusCode': 200,
            "headers": {
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Allow-Methods": "OPTIONS,POST"
                       },
            'body': json.dumps({'message': 'Logout successful'})
        }
    
    except Exception as e:
        print(f"Logout error: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({'message': 'Error during logout.'})
        }

def forgot_password(event):
    """Teachers/students can't reset their own password - this files a request for an admin.

    Always answers the same way so the endpoint can't be used to find out which
    emails have accounts.
    """
    generic = respond(200, {'message': 'If that account exists, your password reset request has been sent to the administrator.'})
    try:
        body = json.loads(event.get('body') or '{}')
        email = (body.get('email') or '').strip()
        if not email:
            return respond(400, {'message': 'Email is required.'})

        user = table.get_item(Key={'email': email}).get('Item')
        # Admins reset their own password from their profile page.
        if not user or user.get('userRole') not in ('student', 'teacher'):
            return generic

        from boto3.dynamodb.conditions import Attr
        pending = reset_table.scan(
            FilterExpression=Attr('user_id').eq(user['user_id']) & Attr('status').eq('pending')
        ).get('Items', [])
        if pending:
            return generic

        reset_table.put_item(Item={
            'request_id': str(uuid.uuid4()),
            'user_id': user['user_id'],
            'user_name': user.get('name', ''),
            'user_email': user['email'],
            'role': user['userRole'],
            'requested_at': int(time.time()),
            'status': 'pending',
            'source': 'login'
        })
        return generic
    except Exception as e:
        print(f"Forgot password error: {str(e)}")
        return respond(500, {'message': 'Could not submit your request. Please try again.'})

def change_password(event):
    """Authenticated change. Also clears the must-change flag set by an admin-issued temp password."""
    try:
        user = authenticated_user(event)
        if not user:
            return respond(401, {'message': 'Invalid or expired token'})

        body = json.loads(event.get('body') or '{}')
        current = body.get('currentPassword') or ''
        new = body.get('newPassword') or ''

        if not verify_password(base64.b64decode(user['password']), current):
            return respond(400, {'message': 'Current password is incorrect.'})
        problem = password_problem(new)
        if problem:
            return respond(400, {'message': problem})
        if new == current:
            return respond(400, {'message': 'Choose a password different from your current one.'})

        table.update_item(
            Key={'email': user['email']},
            UpdateExpression='set password = :p, must_change_password = :f, updated_at = :t',
            ExpressionAttributeValues={
                ':p': base64.b64encode(hash_password(new)).decode('utf-8'),
                ':f': False,
                ':t': datetime.now().isoformat()
            }
        )
        return respond(200, {'message': 'Password updated.'})
    except Exception as e:
        print(f"Change password error: {str(e)}")
        return respond(500, {'message': 'Could not update your password. Please try again.'})

def me(event):
    """Fresh copy of the account fields the profile page shows (status/role are admin-managed)."""
    user = authenticated_user(event)
    if not user:
        return respond(401, {'message': 'Invalid or expired token'})
    return respond(200, {
        'user_id': user['user_id'],
        'name': user.get('name', ''),
        'email': user['email'],
        'userRole': user.get('userRole', ''),
        'status': user.get('status', 'active'),
        'must_change_password': bool(user.get('must_change_password', False)),
        'photoUrl': photo_view_url(user)
    })

def profile_photo(event):
    """Two steps: action=upload-url returns a presigned PUT, then action=confirm records it."""
    try:
        user = authenticated_user(event)
        if not user:
            return respond(401, {'message': 'Invalid or expired token'})
        if not PROFILE_PHOTO_BUCKET:
            return respond(503, {'message': 'Profile photos are not configured.'})

        body = json.loads(event.get('body') or '{}')
        s3 = _s3()
        key = f"profile-photos/{user['user_id']}.{PHOTO_TYPES.get(body.get('contentType'), '')}"

        if body.get('action') == 'upload-url':
            if body.get('contentType') not in PHOTO_TYPES:
                return respond(400, {'message': 'Use a JPG, PNG or WebP image.'})
            size = body.get('size')
            if isinstance(size, bool) or not isinstance(size, (int, float)) or not 0 < size <= PHOTO_MAX_BYTES:
                return respond(400, {'message': 'Photo must be under 5 MB.'})
            # The size is part of the signature: S3 refuses an upload of any other length.
            upload_url = s3.generate_presigned_url(
                'put_object',
                Params={'Bucket': PROFILE_PHOTO_BUCKET, 'Key': key, 'ContentType': body['contentType'], 'ContentLength': int(size)},
                ExpiresIn=300)
            return respond(200, {'uploadUrl': upload_url, 'key': key})

        if body.get('action') == 'confirm':
            # Only accept the key we issued for this user, and only once the object exists.
            if body.get('contentType') not in PHOTO_TYPES or body.get('key') != key:
                return respond(400, {'message': 'Invalid photo key.'})
            head = s3.head_object(Bucket=PROFILE_PHOTO_BUCKET, Key=key)
            if head['ContentLength'] > PHOTO_MAX_BYTES:
                s3.delete_object(Bucket=PROFILE_PHOTO_BUCKET, Key=key)
                return respond(400, {'message': 'Photo must be under 5 MB.'})
            table.update_item(
                Key={'email': user['email']},
                UpdateExpression='set photo_key = :k',
                ExpressionAttributeValues={':k': key})
            return respond(200, {'photoUrl': photo_view_url({'photo_key': key})})

        return respond(400, {'message': 'Unknown action.'})
    except Exception as e:
        print(f"Profile photo error: {str(e)}")
        return respond(500, {'message': 'Could not update your photo. Please try again.'})

def lambda_handler(event, context):
    """Routes the request; gradeant_common.http.handle adds CORS to every response and
    answers OPTIONS, so the browser always sees error messages (e.g. "Invalid credentials")."""
    return handle(event, context, _route, key='message')


def _route(event):
    """Picks the handler for the request's path (config/app.yaml api.auth) and method."""
    path = event.get('path', '')
    paths = config.api_paths().get('auth') or {}
    base = paths.get('base', '/auth')
    routes = {
        base + paths.get('signup', '/signup'): signup,
        base + paths.get('login', '/login'): login,
        base + paths.get('logout', '/logout'): logout,
        base + paths.get('forgotPassword', '/forgot-password'): forgot_password,
        base + paths.get('changePassword', '/change-password'): change_password,
        base + paths.get('me', '/me'): me,
        base + paths.get('profilePhoto', '/profile-photo'): profile_photo,
    }
    route = routes.get(path)
    if route is None or event.get('httpMethod', 'POST') != 'POST':
        return respond(404, {'message': 'Not Found'})
    return route(event)
