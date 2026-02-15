"""py2app build configuration for Sticker Sheet Maker.

Usage (macOS only):
    python setup.py py2app

Produces: dist/Sticker Sheet.app
"""
from setuptools import setup

APP = ['sticker_app.py']
DATA_FILES = []
OPTIONS = {
    'argv_emulation': False,  # must be False for Qt apps
    'iconfile': 'icons/StickerSheet.icns',
    'plist': {
        'CFBundleName': 'Sticker Sheet',
        'CFBundleDisplayName': 'Sticker Sheet',
        'CFBundleIdentifier': 'com.stickersheet.app',
        'CFBundleVersion': '1.0.0',
        'CFBundleShortVersionString': '1.0',
        'LSMinimumSystemVersion': '11.0',
        'NSHighResolutionCapable': True,
        'CFBundleDocumentTypes': [{
            'CFBundleTypeName': 'Sticker Sheet Project',
            'CFBundleTypeExtensions': ['sticker'],
            'CFBundleTypeIconFile': 'StickerDoc',
            'CFBundleTypeRole': 'Editor',
            'LSHandlerRank': 'Owner',
            'LSItemContentTypes': ['com.stickersheet.sticker'],
        }],
        'UTExportedTypeDeclarations': [{
            'UTTypeIdentifier': 'com.stickersheet.sticker',
            'UTTypeDescription': 'Sticker Sheet Project',
            'UTTypeConformsTo': ['public.data'],
            'UTTypeTagSpecification': {
                'public.filename-extension': ['sticker'],
            },
        }],
    },
    'packages': ['PySide6', 'PIL'],
    'strip': False,  # avoid "Operation not permitted" on macOS SIP-protected binaries
}

setup(
    app=APP,
    data_files=DATA_FILES,
    options={'py2app': OPTIONS},
    setup_requires=['py2app'],
)
