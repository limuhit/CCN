chcp 65001
call.\kill_tasks.bat
echo on
set PATH=%PATH%;D:\Program Files\vs2017\Microsoft Visual Studio\2019\Community\VC\Tools\MSVC\14.25.28610\bin\Hostx64\x64;
rem echo %PATH%
python setup.py install > log.txt
python .\filter_error.py 