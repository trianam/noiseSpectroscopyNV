#     notifier.py
#     Utility functions to send a mail programmatically (e.g. to notify when the training finish).
#     Copyright (C) 2021  Stefano Martina (stefano.martina@unifi.it)
#
#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.
#
#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.
#
#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import smtplib
from email.mime.text import MIMEText
from notifierConfig import account

def sendMessage(subject, message=''):
	msg = MIMEText(message)

	msg['Subject'] = subject
	msg['From'] = account.fromMail
	msg['To'] = account.toMail

	smtpObj = smtplib.SMTP_SSL(account.smtpServer, account.smtpPort)
	smtpObj.login(account.userName, account.password)
	smtpObj.send_message(msg)
	smtpObj.quit()

def sendFile(subject, fileName, message=''):
	with open(fileName) as fp:
		msg = MIMEText(message+"\n"+fp.read())

	msg['Subject'] = subject
	msg['From'] = account.fromMail
	msg['To'] = account.toMail

	smtpObj = smtplib.SMTP_SSL(account.smtpServer, account.smtpPort)
	smtpObj.login(account.userName, account.password)
	smtpObj.send_message(msg)
	smtpObj.quit()

