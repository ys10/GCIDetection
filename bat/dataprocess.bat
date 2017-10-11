hed -n temp.sfs
set DestPath=C:\Users\Silver\Desktop\AW\
rem egg
set DestExt=*.egg 
for /f "delims=" %%i   in ('dir  /b/a-d/s  %DestPath%\%DestExt%')  do (
	slink -ilx. -tWAV %%i temp.sfs
	hqtx temp.sfs
	txlist -c -o %%~ni.marks temp.sfs
	remove -e temp.sfs
)
pause