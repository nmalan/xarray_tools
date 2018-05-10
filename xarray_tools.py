from __future__ import print_function

def xarray_trend(xarr):
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


def corr_vars(xarr1, xarr2):
    from pandas import DataFrame
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


def create_cartopy_axes(ax=None, lw=0.5, c='k'):
    """
    USAGE:      xarray.DataArray2D.plot(**cartopy_axes_spstere())
    OUTPUT:     south polar steriographic plot
    REQUIRES:   xarray, cartopy
    """

    import cartopy.crs as ccrs
    import cartopy.feature
    import matplotlib.path as mpath
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        ax = plt.axes(projection=ccrs.SouthPolarStereo())

    theta = np.linspace(0, 2 * np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)

    ax.set_boundary(circle, transform=ax.transAxes)

    ax.add_feature(cartopy.feature.LAND, facecolor='#CCCCCC', zorder=5)
    ax.add_feature(cartopy.feature.COASTLINE, linewidth=lw, zorder=5)
    ax.add_feature(cartopy.feature.BORDERS, linewidth=lw)

    ca = dict(robust=True, ax=ax, transform=ccrs.PlateCarree())

    return ca


def regrid_unweighted(xds, lat_bins, lon_bins, how='mean'):
    """
    Regrid an xr.DataArray with unweighted averaging.
    The input requires bins for lat and lon.
    The center points are calculated from the bins
    """

    attrs = {}
    for key in xds.data_vars:
        attrs[key] = xds[key].attrs

    lat_ctr = (lat_bins[1:] + lat_bins[:-1]) / 2
    lon_ctr = (lon_bins[1:] + lon_bins[:-1]) / 2

    lon_reg = xds.groupby_bins('lon', lon_bins, labels=lon_ctr)
    lon_reg = getattr(lon_reg, how)('lon')

    lat_reg = lon_reg.groupby_bins('lat', lat_bins, labels=lat_ctr)
    lat_reg = getattr(lat_reg, how)('lat')

    regridd = lat_reg.rename({'lon_bins': 'lon', 'lat_bins': 'lat'})
    regridd = regridd.transpose('time', 'lat', 'lon')

    for key in regridd.data_vars:
        regridd[key].attrs = dict(attrs[key])

    return regridd


def read_netcdfs(files, dim, transform_func=None, verbose=False):
    import xarray as xr
    from glob import glob

    def process_one_path(path, v=False):
        # use a context manager, to ensure the file gets closed after use
        if v:
            print (path)
        with xr.open_dataset(path) as ds:
            # transform_func should do some sort of selection or
            # aggregation
            if transform_func is not None:
                ds = transform_func(ds)
            # load all data from the transformed dataset, to ensure we can
            # use it after closing each original file
            ds.load()
            return ds

    paths = sorted(glob(files))
    datasets = [process_one_path(p, v=verbose) for p in paths]
    combined = xr.concat(datasets, dim)
    return combined


def resample_day2mon_deg2qrt(fname, sname):
    import numpy as np

    xb = np.arange(-180, 180.1, 1)
    yb = np.arange( -90,  90.1, 1)

    print (fname, end=': ')
    xds = read_netcdfs(fname, 'time', verbose=False)

    print ('resampling', end=', ')
    xdsM = xds.resample('1MS', 'time', keep_attrs=True)
    xds1 = regrid_unweighted(xdsM, yb, xb, how='mean')

    for key in xds1.data_vars:
        xds[key].encoding = {'complevel': 4, 'zlib': True}

    print ('saving')
    xds1.to_netcdf(sname)

    xds.close()
    xdsM.close()
    xds1.close()
