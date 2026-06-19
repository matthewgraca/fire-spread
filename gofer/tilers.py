import cartopy.io.img_tiles as cimgt

class CartoDBTiles(cimgt.GoogleWTS):
    def __init__(self, style="light_all", cache=False):
        super().__init__(cache=cache)
        self.style = self._validate_style(style)

    def _image_url(self, tile):
        x, y, z = tile
        return (
            "https://cartodb-basemaps-a.global.ssl.fastly.net/"
            f"{self.style}/{z}/{x}/{y}.png"
        )

    def _validate_style(self, style):
        valid_styles = [
            'light_all',
            'light_nolabels',
            'light_only_labels',
            'dark_all',
            'dark_nolabels',
            'dark_only_labels',
            'rastertiles/voyager',
            'rastertiles/voyager_nolabels',
            'rastertiles/voyager_only_labels',
            'rastertiles/voyager_labels_under'
        ]
        if style not in valid_styles:
            raise ValueError(f'style must be {valid_styles}')

        return style
