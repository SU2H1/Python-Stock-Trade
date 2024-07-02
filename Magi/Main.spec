# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all, copy_metadata

block_cipher = None

datas = []
binaries = []
hiddenimports = []

# Add necessary metadata
datas += copy_metadata('transformers')
datas += copy_metadata('tokenizers')
datas += copy_metadata('filelock')
datas += copy_metadata('requests')
datas += copy_metadata('packaging')
datas += copy_metadata('regex')
datas += copy_metadata('tqdm')
datas += copy_metadata('numpy')
datas += copy_metadata('torch')

# Collect all necessary files for transformers
datas_t, binaries_t, hiddenimports_t = collect_all('transformers')
datas += datas_t
binaries += binaries_t
hiddenimports += hiddenimports_t

a = Analysis(
    ['Main.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=['transformers', 'tqdm', 'regex', 'requests', 'packaging', 'filelock', 'numpy', 'tokenizers', 'torch'] + hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='Main',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
