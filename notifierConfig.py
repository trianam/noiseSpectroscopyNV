#     notifierConfigTEMPLATE.py
#     Template for the configuration file for the notifier (copy in notifierConfig.py and modify with your account data).
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

from conf import Conf

account = Conf({
    'fromMail':         "xxx@yyy.zz",
    'toMail':           "xxx@yyy.zz",
    'smtpServer':       "xxx.yy",
    'smtpPort':         25,
    'userName':         "xxx",
    'password':         "xxx",
})

