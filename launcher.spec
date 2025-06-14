# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

# Collect all required files
added_files = [
    ('backend.py', '.'),
    ('frontend.py', '.'),
    ('database.py', '.'),
    ('rain_utils.py', '.'),
    ('models/*.pt', 'models'),  # Only include .pt files
    ('assets/icons/*.png', 'assets/icons'),  # Only include .png files
    ('backlog/icons/*.png', 'backlog/icons'),
]

a = Analysis(
    ['launcher.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('backend.py', '.'),
        ('frontend.py', '.'),
        ('database.py', '.'),
        ('rain_utils.py', '.'),
        ('models/*', 'models'),
        ('assets/icons/*', 'assets/icons'),
        ('backlog/icons/*', 'backlog/icons')
    ],
    hiddenimports=[
        'win32api', 'win32event', 'win32process', 'winerror',
        'cv2', 'numpy', 'torch', 'torchvision',
        'PIL', 'PIL._tkinter_finder',
        'requests', 'customtkinter',
        'sqlite3', 'passlib.hash',
        'psutil', 'colorama',
        'facenet_pytorch',
        'uvicorn', 'fastapi', 'qdrant_client',
        'starlette', 'asyncio',
        'logging', 'datetime'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='FaceRecognition',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # Changed to True to show console
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='backlog/icons/id.png'
)
