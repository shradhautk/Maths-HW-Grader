"""Outgoing email (backend/src/aws/ses.js): Amazon SES in production, logged locally.

MAIL_DRIVER=ses|log (default: ses in production, log otherwise).
"""
import json

import boto3

from . import config

sent_mail = []  # MAIL_DRIVER=log only: inspected by tests


def driver():
    return config.env('MAIL_DRIVER', 'ses' if config.is_production() else 'log')


def send_email(*, sender, to, subject, text=None, html=None, reply_to=None):
    if driver() == 'log':
        sent_mail.append({'from': sender, 'to': to, 'subject': subject, 'text': text, 'html': html, 'replyTo': reply_to})
        print(json.dumps({'level': 'info', 'msg': 'email (not sent: MAIL_DRIVER=log)', 'to': to, 'subject': subject}))
        return
    body = {'Html': {'Data': html}} if html else {'Text': {'Data': text or ''}}
    args = {
        'Source': sender,
        'Destination': {'ToAddresses': [to]},
        'Message': {'Subject': {'Data': subject}, 'Body': body},
    }
    if reply_to:
        args['ReplyToAddresses'] = [reply_to]
    boto3.client('ses', region_name=config.data_region()).send_email(**args)
