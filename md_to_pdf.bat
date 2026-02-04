@echo off
cd /d "%~dp0"
set "MIKTEX=%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64"
set "PATH=%MIKTEX%;%PATH%"
echo Converting RCDT_bioRxiv_manuscript.md to PDF...
pandoc RCDT_bioRxiv_manuscript.md -o RCDT_bioRxiv_manuscript.pdf --pdf-engine=xelatex -V geometry:margin=1in -V fontsize=12pt
if %errorlevel% equ 0 (
    echo Success! PDF saved as RCDT_bioRxiv_manuscript.pdf
) else (
    echo Conversion failed. Try: pandoc RCDT_bioRxiv_manuscript.md -o RCDT_bioRxiv_manuscript.pdf --pdf-engine=pdflatex -V geometry:margin=1in -V fontsize=12pt
)
pause
