"""pyvistra plugin entry point for jssl_denoise.

Declared in ``pyproject.toml``::

    [project.entry-points."pyvistra.plugins"]
    jssl-denoise = "jssl_denoise._pyvistra_plugin:register"

Discovered lazily by pyvistra the first time any viewer builds a menu bar
(see ``pyvistra.plugins.discover_plugins``) -- only imports Qt/torch at
that point, never at ``import jssl_denoise`` time.
"""


def _show_denoise_dialog(window):
    from .pyvistra_gui.denoise_dialog import DenoiseDialog

    dlg = getattr(window, "_jssl_denoise_dialog", None)
    if dlg is None:
        dlg = DenoiseDialog(viewer=window, parent=window)
        window._jssl_denoise_dialog = dlg
    dlg.show()
    dlg.raise_()


def register():
    from pyvistra.plugins import add_menu_item
    from pyvistra.ui.window import ImageWindow

    ImageWindow._jssl_denoise_show_dialog = _show_denoise_dialog
    add_menu_item(
        ImageWindow,
        "Image",
        {
            "label": "Denoise...",
            "method": "_jssl_denoise_show_dialog",
            "tooltip": "Self-supervised blind denoising: train a D-Net/N-Net pair on this stack, then denoise.",
        },
        submenu="Denoising",
    )
