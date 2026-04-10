@echo off
title StatsRunning - Serveur Flask
echo ----------------------------------------------------------
echo   LANCEMENT DE STATSRUNNING (MARATHON DE NANTES)
echo ----------------------------------------------------------

:: On se place dans le dossier du script .bat (Racine)
cd /d "%~dp0"

:: 1. Activation de l'environnement Python standard
echo [1/2] Activation de l'environnement marathon-env...
call marathon-env\Scripts\activate.bat

:: 2. Lancement du serveur
echo [2/2] Demarrage du serveur Flask...
echo.

if exist "main.py" (
    python main.py
) else if exist "app\main.py" (
    cd app
    python main.py
) else (
    echo [ERREUR] Impossible de trouver main.py. 
)

pause