chcp 65001
pyinstaller main.py
copy "Launch Tower.png " "dist/main"
copy "warn.png" "dist/main"
rd /s /q build
xcopy "dist/main" "直播室链路选择程序" /s /i
rd /s /q dist