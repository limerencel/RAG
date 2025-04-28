@echo off
echo Installing RAGChat CLI tool...
pip install -e .

if %ERRORLEVEL% equ 0 (
    echo Installation successful!
    echo You can now use the 'ragchat' command from your terminal.
    echo.
    echo Example usage:
    echo   ragchat --dir C:\path\to\documents
    echo   ragchat --files doc1.txt doc2.pdf
    echo.
    echo Don't forget to create a .env file with your xAI API key:
    echo XAI_API_KEY=your-api-key-here
) else (
    echo Installation failed. Please check the error messages above.
    echo Try using: pip install --upgrade pip
    echo Then run the install script again.
)

pause 