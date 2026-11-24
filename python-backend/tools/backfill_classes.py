"""One-off: give existing students their class (Users.classes) before deploying the
security fix that hides other classes' work from students with no classes.

Accounts created from an admin invite before the fix have no Users.classes, but the
admin console recorded the invite's class as GradeAnt_Users.school. This copies it.
Students who signed themselves up have no invite, so they are listed for an admin to
place in a class by hand.

    python tools/backfill_classes.py --region eu-north-1            # dry run: shows what would change
    python tools/backfill_classes.py --region eu-north-1 --apply    # writes Users.classes

Needs the AWS CLI credentials of an account that can Scan Users and GradeAnt_Users and
UpdateItem on Users. Table names follow USERS_TABLE / ADMIN_USERS_TABLE (defaults
Users / GradeAnt_Users). Safe to run twice: it only touches students with no classes.
"""
import argparse
import os

import boto3
from boto3.dynamodb.conditions import Attr


def scan_all(table, **kwargs):
    items, start = [], None
    while True:
        if start:
            kwargs['ExclusiveStartKey'] = start
        page = table.scan(**kwargs)
        items.extend(page.get('Items', []))
        start = page.get('LastEvaluatedKey')
        if not start:
            return items


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--region', required=True)
    ap.add_argument('--apply', action='store_true', help='write the changes (default: dry run)')
    args = ap.parse_args()

    ddb = boto3.resource('dynamodb', region_name=args.region)
    users = ddb.Table(os.environ.get('USERS_TABLE', 'Users'))
    admin_users = ddb.Table(os.environ.get('ADMIN_USERS_TABLE', 'GradeAnt_Users'))

    students = scan_all(users, FilterExpression=Attr('userRole').eq('student') & (Attr('classes').not_exists() | Attr('classes').size().eq(0)))
    invited = {u['user_id']: u for u in scan_all(admin_users) if u.get('user_id')}

    fixed, unplaced = 0, []
    for s in sorted(students, key=lambda x: x.get('email', '')):
        cls = str((invited.get(s.get('user_id')) or {}).get('school') or '').strip()
        if not cls:
            unplaced.append(s.get('email'))
            continue
        print(f"{'set' if args.apply else 'would set'} classes=[{cls!r}] for {s.get('email')}")
        if args.apply:
            users.update_item(
                Key={'email': s['email']},
                UpdateExpression='SET classes = :c',
                ConditionExpression=Attr('classes').not_exists() | Attr('classes').size().eq(0),
                ExpressionAttributeValues={':c': [cls]},
            )
        fixed += 1

    print(f"\n{len(students)} student(s) without classes: {fixed} {'updated' if args.apply else 'can be updated'} from their invite.")
    if unplaced:
        print(f'{len(unplaced)} have no invite class (self sign-ups or invites without a class); place them in a class by hand:')
        for email in unplaced:
            print(f'  {email}')
    if not args.apply and fixed:
        print('\nDry run only. Re-run with --apply to write.')


if __name__ == '__main__':
    main()
