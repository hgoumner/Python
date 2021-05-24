@ECHO off
set IPADDRESS=8.8.8.8
set INTERVAL=60
set logfile="C:\Users\gou\Desktop\ping_check.log"

for /f "tokens=*" %%A in ('ping %IPADDRESS% -n 1 ') do (echo %%A>>%logfile% && GOTO Ping)
:Ping
for /f "tokens=* skip=2" %%A in ('ping %IPADDRESS% -n 1 ') do (
    echo %date% %time:~0,2%:%time:~3,2%:%time:~6,2% %%A>>%logfile%
    echo %date% %time:~0,2%:%time:~3,2%:%time:~6,2% %%A
    timeout 1 >NUL 
 TIMEOUT %INTERVAL%
    GOTO Ping)
