"""Create GradeAnt's first admin account in AWS DynamoDB.

Nobody can sign up as an admin, so the first one is created with this script.
After that, admins invite everyone else from the admin console.

It writes two records with the same user_id, in one transaction (both or neither):

  Users           (key: email)    the login account /auth/login checks, with userRole "admin"
  GradeAnt_Users  (key: user_id)  the admin console's user list, with role "admin", status "active"

It never overwrites anything. If the email is already in Users, or the user_id is
already in GradeAnt_Users, nothing is written.

Usage (needs AWS credentials allowed to write both tables, e.g. after `aws configure`):

  pip install boto3
  python admin/create_first_admin.py --email admin@yourschool.org --name "Asha Patel"
  python admin/create_first_admin.py --email ... --name ... --dry-run   # show the records, write nothing

The password is asked for at the prompt (it is never a command-line argument, so it
stays out of your shell history). Log in afterwards on the login page with the
"Admin" role selected.
"""

import argparse
import base64
import getpass
import hashlib
import os
import sys
import uuid
from datetime import datetime


def hash_password_for_login(password: str) -> str:
    """The same format authetication.py's login checks:
    base64(32-byte salt + PBKDF2-HMAC-SHA256(password, salt, 100,000 rounds))."""
    salt = os.urandom(32)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return base64.b64encode(salt + digest).decode("utf-8")


def password_problem(password: str):
    """The app's password rules: 8+ characters, an uppercase letter, a number."""
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if not any(c.isupper() for c in password):
        return "Password must include an uppercase letter."
    if not any(c.isdigit() for c in password):
        return "Password must include a number."
    return None


def ask_password() -> str:
    while True:
        password = getpass.getpass("Password for the new admin: ")
        problem = password_problem(password)
        if problem:
            print(f"  {problem} Try again.")
            continue
        if getpass.getpass("Type it again: ") != password:
            print("  The passwords don't match. Try again.")
            continue
        return password


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the first GradeAnt admin in DynamoDB.")
    parser.add_argument("--email", required=True, help="the admin's login email (stored exactly as typed)")
    parser.add_argument("--name", required=True, help="the admin's full name")
    parser.add_argument("--region", default=os.environ.get("AWS_REGION", "eu-north-1"),
                        help="AWS region of the tables (default: eu-north-1)")
    parser.add_argument("--login-table", default=os.environ.get("LOGIN_USERS_TABLE", "Users"))
    parser.add_argument("--admin-table", default=os.environ.get("USERS_TABLE", "GradeAnt_Users"))
    parser.add_argument("--profile", help="AWS CLI profile to use")
    parser.add_argument("--dry-run", action="store_true", help="print the records without writing them")
    args = parser.parse_args()

    email = args.email.strip()
    name = args.name.strip()
    if "@" not in email or " " in email:
        print(f"'{email}' doesn't look like an email address.", file=sys.stderr)
        return 2
    if not name:
        print("--name can't be empty.", file=sys.stderr)
        return 2

    password = ask_password()
    user_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat()

    login_item = {
        "email": email,
        "user_id": user_id,
        "name": name,
        "password": hash_password_for_login(password),
        "userRole": "admin",
        "status": "active",
        "created_at": now,
        "updated_at": now,
    }
    admin_item = {
        "user_id": user_id,
        "email": email,
        "name": name,
        "role": "admin",
        "school": "",
        "status": "active",
        "created_at": now,
    }

    if args.dry_run:
        shown = dict(login_item, password="<hash, not shown>")
        print(f"Dry run: nothing written.\n\n{args.login_table}:\n  {shown}\n\n{args.admin_table}:\n  {admin_item}")
        return 0

    try:
        import boto3
        from boto3.dynamodb.types import TypeSerializer
        from botocore.exceptions import ClientError, NoCredentialsError
    except ImportError:
        print("boto3 isn't installed. Run: pip install boto3", file=sys.stderr)
        return 1

    session = boto3.Session(profile_name=args.profile, region_name=args.region)
    client = session.client("dynamodb")
    ser = TypeSerializer()

    def typed(item):
        return {k: ser.serialize(v) for k, v in item.items()}

    try:
        client.transact_write_items(TransactItems=[
            {"Put": {
                "TableName": args.login_table,
                "Item": typed(login_item),
                "ConditionExpression": "attribute_not_exists(email)",
            }},
            {"Put": {
                "TableName": args.admin_table,
                "Item": typed(admin_item),
                "ConditionExpression": "attribute_not_exists(user_id)",
            }},
        ])
    except NoCredentialsError:
        print("No AWS credentials found. Run `aws configure` (or pass --profile) and try again.", file=sys.stderr)
        return 1
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code == "TransactionCanceledException":
            reasons = [r.get("Code") for r in e.response.get("CancellationReasons", [])]
            if reasons and reasons[0] == "ConditionalCheckFailed":
                print(f"An account with {email} already exists in {args.login_table}. Nothing was written.",
                      file=sys.stderr)
                return 1
            print(f"DynamoDB cancelled the write ({reasons}). Nothing was written.", file=sys.stderr)
            return 1
        if code == "ResourceNotFoundException":
            print(f"Table not found in {args.region}: check that {args.login_table} and {args.admin_table} "
                  f"exist there (or pass --region / --login-table / --admin-table).", file=sys.stderr)
            return 1
        if code == "AccessDeniedException":
            print(f"Your AWS identity can't write these tables: {e.response['Error'].get('Message')}",
                  file=sys.stderr)
            return 1
        raise

    print(f"Admin created: {name} <{email}> (user_id {user_id}) in {args.region}.")
    print("Log in on the login page with the Admin role selected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
