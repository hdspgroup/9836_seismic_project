# -*- mode: python ; coding: utf-8 -*-


block_cipher = None


a = Analysis(
    ['main_window.py'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[],
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

a.datas += [("./assets/icons/main.png", "assets/icons/main.png", "DATA"),
            ("./assets/icons/tuning.png", "assets/icons/tuning.png", "DATA"),
            ("./assets/icons/comparison.png", "assets/icons/comparison.png", "DATA"),
            ("./assets/icons/view.png", "assets/icons/view.png", "DATA"),
            ("./assets/icons/save.png", "assets/icons/save.png", "DATA"),
            ("./assets/icons/run.png", "assets/icons/run.png", "DATA"),
            ("./assets/icons/report.png", "assets/icons/report.png", "DATA"),
            ("./assets/icons/info.png", "assets/icons/info.png", "DATA"),
            ("./assets/icons/seismic.png", "assets/icons/seismic.png", "DATA"),
            ("./assets/icons/report.png", "assets/icons/report.png", "DATA"),
            ("./assets/parameters/lambda_init.png", "assets/parameters/lambda_init.png", "DATA"),
            ("./assets/parameters/lambda_end.png", "assets/parameters/lambda_end.png", "DATA"),
            ("./assets/parameters/mu_init.png", "assets/parameters/mu_init.png", "DATA"),
            ("./assets/parameters/mu_end.png", "assets/parameters/mu_end.png", "DATA"),
            ("./assets/parameters/rho_init.png", "assets/parameters/rho_init.png", "DATA"),
            ("./assets/parameters/rho_end.png", "assets/parameters/rho_end.png", "DATA"),
            ("./assets/parameters/gamma_init.png", "assets/parameters/gamma_init.png", "DATA"),
            ("./assets/parameters/gamma_end.png", "assets/parameters/gamma_end.png", "DATA"),
            ("./assets/parameters/alpha_init.png", "assets/parameters/alpha_init.png", "DATA"),
            ("./assets/parameters/alpha_end.png", "assets/parameters/alpha_end.png", "DATA"),
            ("./assets/parameters/beta_init.png", "assets/parameters/beta_init.png", "DATA"),
            ("./assets/parameters/beta_end.png", "assets/parameters/beta_end.png", "DATA"),
            ("./assets/parameters/lambda.png", "assets/parameters/lambda.png", "DATA"),
            ("./assets/parameters/mu.png", "assets/parameters/mu.png", "DATA"),
            ("./assets/parameters/rho.png", "assets/parameters/rho.png", "DATA"),
            ("./assets/parameters/gamma.png", "assets/parameters/gamma.png", "DATA"),
            ("./assets/parameters/alpha.png", "assets/parameters/alpha.png", "DATA"),
            ("./assets/parameters/beta.png", "assets/parameters/beta.png", "DATA"),]

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='main_window',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='main_window',
)
