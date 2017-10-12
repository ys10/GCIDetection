hed -n temp.sfs
set DestPath=C:\Users\Silver\Desktop\AW\
rem egg
set DestExt=*.egg 
for /f "delims=" %%i   in ('dir  /b/a-d/s  %DestPath%\%DestExt%')  do (
	COPY %%i %%~ni.egg
)
pause