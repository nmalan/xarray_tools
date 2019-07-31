from __future__ import print_function
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt


@xr.register_dataarray_accessor('stats')
class Statistics(object):
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def pca_decomp(self, n_components=10, return_plots=False, return_pca=False):
        """
        Apply a principle component decomposition to a dataset with
        time, lat, lon axes.
        """
        from sklearn.decomposition import PCA

        def unnan(arr):
            t, y, x = arr.shape
            flat = arr.reshape(t, -1)
            mask = ~np.isnan(flat).any(0)
            return flat[:, mask], mask

        def renan(arr, mask, shape=None):
            out = np.ndarray([min(arr.shape), mask.size]) * np.NaN
            if np.argmin(arr.shape) == 1:
                arr = arr.T
            out[:, mask] = arr
            out = out
            if shape:
                out = out.reshape(*shape)
            return out

        xda = self._obj
        t, y, x = xda.dims

        assert t.lower() in ['time', 'date', 'tmnth', 'days'], 'DataArray needs to have time as first dimension'
        assert y.lower() in 'ylatitude', 'DataArray needs to have latitude as second dimension'
        assert x.lower() in 'xlongitude', 'DataArray needs to have longitude as third dimension'

        coords = {d: xda[d].values for d in xda.dims}
        coords.update({'n_components': np.arange(n_components)})

        pca = PCA(n_components=n_components)

        v, m = unnan(xda.values)

        trans = pca.fit_transform(v.T)
        trans_3D = renan(trans, m, shape=[n_components, coords[y].size, coords[x].size])

        xds = xr.Dataset(attrs={'name': xda.name})
        dims = ['n_components', 'lat', 'lon']
        props = dict(coords={k: coords[k] for k in dims}, dims=dims)
        xds['transformed'] = xr.DataArray(trans_3D, **props)

        dims = ['n_components', 'time']
        props = dict(coords={k: coords[k] for k in dims}, dims=dims)
        xds['principle_components'] = xr.DataArray(pca.components_, **props)

        dims = ['time']
        props = dict(coords={k: coords[k] for k in dims}, dims=dims)
        xds['mean_'] = xr.DataArray(pca.mean_, **props)

        dims = ['n_components']
        props = dict(coords={k: coords[k] for k in dims}, dims=dims)
        xds['variance_explained'] = xr.DataArray(pca.explained_variance_ratio_, **props)

        if return_plots and return_pca:
            fig = self.pca_plot(xds)
            return xds, pca, fig
        elif return_plots:
            fig = self.pca_plot(xds)
            return xds, fig
        elif return_pca:
            return xds, pca
        else:
            return xds

    @staticmethod
    def pca_plot(xds_pca):
        n = xds_pca.n_components.size
        fig = plt.figure(figsize=[15, n * 3.2], dpi=120)
        shape = n, 5
        ax = []

        for i in range(shape[0]):
            ax += [plt.subplot2grid(shape, [i, 0], colspan=3, fig=fig),
                   plt.subplot2grid(shape, [i, 3], colspan=2, fig=fig, facecolor='#AAAAAA')],

        t = xds_pca.principle_components.dims[-1]
        y, x = xds_pca.transformed.dims[1:]
        for i in xds_pca.n_components.values:
            pt = xds_pca[t].values
            px = xds_pca[x].values
            py = xds_pca[y].values
            pz = xds_pca.transformed[i].to_masked_array()

            var = xds_pca.variance_explained[i].values * 100
            lim = np.nanpercentile(abs(pz.filled(np.nan)), 99)

            a0 = ax[i][0]
            a1 = ax[i][1]

            a0.plot(pt, xds_pca.principle_components[i].values)
            a0.axhline(0, color='k')
            a0.set_ylabel('Component {}\n({:.2f}%)'.format(
                i+1, var), fontsize=12)

            img = a1.pcolormesh(px, py, pz, vmin=-lim,
                                vmax=lim, cmap=plt.cm.RdBu_r)
            plt.colorbar(img, ax=a1)
            img.colorbar.set_label('Transformed units')

            if i != (shape[0]-1):
                a0.set_xticklabels([])
                a1.set_xticklabels([])
            else:
                pass

        title = "Principle Component Analysis (PCA) for {} showing the first {} components"
        fig.suptitle(title.format(xds_pca.name, n), y=1.01,
                     x=0.5, fontsize=16, fontweight='bold')
        fig.tight_layout()

        return fig

    def trend(self):
        """
        Calculates the trend of the data along the 'time' dimension
        of the input array (xarr).
        USAGE:  x_DS = xarray_trend(xarr)
        INPUT:  xarr is an xarray DataArray with dims:
                    time, [lat, lon]
                    where lat and/or lon are optional
        OUTPUT: xArray Dataset with:
                    original xarr input
                    slope
                    p-value

        TODO?
        There could be speed improvements (using numpy at the moment)
        """

        from scipy import stats
        # getting shapes

        xarr = self._obj

        n = xarr.shape[0]

        # creating x and y variables for linear regression
        x = xarr.time.to_pandas().index.to_julian_date().values[:, None]
        y = xarr.to_masked_array().reshape(n, -1)

        # ############################ #
        # LINEAR REGRESSION DONE BELOW #
        xm = x.mean(0)  # mean
        ym = y.mean(0)  # mean
        ya = y - ym  # anomaly
        xa = x - xm  # anomaly

        # variance and covariances
        xss = (xa ** 2).sum(0) / (n - 1)  # variance of x (with df as n-1)
        yss = (ya ** 2).sum(0) / (n - 1)  # variance of y (with df as n-1)
        xys = (xa * ya).sum(0) / (n - 1)  # covariance (with df as n-1)
        # slope and intercept
        slope = xys / xss
        # statistics about fit
        df = n - 2
        r = xys / (xss * yss)**0.5
        t = r * (df / ((1 - r) * (1 + r)))**0.5
        p = stats.distributions.t.sf(abs(t), df)

        # misclaneous additional functions
        # intercept = ym - (slope * xm)
        # yhat = dot(x, slope[None]) + intercept
        # sse = ((yhat - y)**2).sum(0) / (n - 2)  # n-2 is df
        # se = ((1 - r**2) * yss / xss / df)**0.5

        # preparing outputs
        out = xarr.to_dataset(name=xarr.name)
        # first create variable for slope and adjust meta
        out['slope'] = xarr[:2].mean('time').copy()
        out['slope'].name += '_slope'
        out['slope'].attrs['units'] = 'units / day'
        out['slope'].values = slope.reshape(xarr.shape[1:])
        # do the same for the p value
        out['pval'] = xarr[:2].mean('time').copy()
        out['pval'].name += '_Pvalue'
        out['pval'].values = p.reshape(xarr.shape[1:])
        out['pval'].attrs['info'] = ("If p < 0.05 then the results "
                                    "from 'slope' are significant.")

        return out


    def corr_vars(self, xarr2):
        from pandas import DataFrame

        xarr1 = self._obj.copy()
        assert xarr1.shape == xarr2.shape, 'The input DataArray must be the same size as {}'.format(self.name)

        xarr3 = xarr1[:1].mean('time').copy()

        t, y, x = xarr1.shape

        df1 = DataFrame(xarr1.values.reshape(t, y * x))
        df2 = DataFrame(xarr2.values.reshape(t, y * x))

        dfcor = df1.corrwith(df2).values.reshape(y, x)
        xarr3.values = dfcor

        xarr3.attrs['long_name'] = 'Correlation of %s and %s' % (xarr1.name, xarr2.name)
        xarr3.name = 'corr_%s_vs_%s' % (xarr1.name, xarr2.name)

        xarr3.encoding.update({'zlib': True, 'shuffle': True, 'complevel': 4})

        return xarr3


@xr.register_dataarray_accessor('plot_map')
class CartopyMap(object):
    """
    Plot the given 2D array on a cartopy axes (assuming that Lat and Lon exist)
    The default projection is PlateCarree, but can be:
        cartopy.crs.<ProjectionName>()

    If the projection is Stereographic the plot will be round unless
    the keyword arguement `round` is set False.

    If you would like to create a figure with multiple subplots
    you can pass an axes object to the function with keyword argument `ax,
    BUT then you need to specify the projection when you create the axes:
        plt.axes([x0, y0, w, h], projection=cartopy.crs.<ProjectionName>())

    Additional keywords can be given to the function as you would to
    the xr.DataArray.plot function. The only difference is that `robust`
    is set to True by default.

    The function returns a GeoAxes object to which features can be added with:
        ax.add_feature(feature.<FeatureName>, **kwargs)
    By default, LAND and COASTLINE are added, but can be removed by
    setting default_features=False
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, ax=None, proj=None, round=True, default_features=True, **kwargs):
        return self._cartopy(ax=ax, proj=proj, round=round, default_features=default_features, **kwargs)

    def _cartopy(self, ax=None, proj=None, round=True, default_features=True, **kwargs):
        import matplotlib.path as mpath
        import matplotlib.pyplot as plt
        from cartopy import feature
        from cartopy import crs

        xda = self._obj
        assert xda.ndim == 2, 'The array must be two dimensional'

        if ax is None:
            tighten = True
            proj = crs.PlateCarree() if proj is None else proj
            fig, ax = plt.subplots(
                1, 1, figsize=[11, 4], dpi=100,
                subplot_kw={'projection': proj})
        else:
            tighten = False

        stereo_maps = (crs.Stereographic,
                       crs.NorthPolarStereo,
                       crs.SouthPolarStereo)
        if isinstance(proj, stereo_maps) & round:
            theta = np.linspace(0, 2 * np.pi, 100)
            center, radius = [0.5, 0.5], 0.5
            verts = np.vstack([np.sin(theta), np.cos(theta)]).T
            circle = mpath.Path(verts * radius + center)

            ax.set_boundary(circle, transform=ax.transAxes)

        if default_features:
            ax.add_feature(feature.LAND, color='#CCCCCC', zorder=4)
            ax.add_feature(feature.COASTLINE, lw=0.5, zorder=4)

        if 'robust' not in kwargs:
            kwargs['robust'] = True
        if ('cbar_kwargs' not in kwargs) & kwargs.get('add_colorbar', True):
            kwargs['cbar_kwargs'] = {'pad': 0.02}

        axm = xda.plot(ax=ax, transform=crs.PlateCarree(), **kwargs)
        if kwargs.get('add_colorbar', True):
            ax.colorbar = axm.colorbar
        if tighten:
            fig.tight_layout()

        return ax
