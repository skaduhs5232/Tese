import os
import numpy as np
from osgeo import gdal

gdal.UseExceptions()


def ReadGeoTiff(filename):
    raster = gdal.Open(str(filename))
    if raster is None:
        raise FileNotFoundError(f"Não foi possível abrir o GeoTIFF: {filename}")

    band = raster.GetRasterBand(1)
    arr = band.ReadAsArray()

    info = {}
    info["RasterYSize"] = raster.RasterYSize
    info["RasterXSize"] = raster.RasterXSize
    info["GeoTransform"] = raster.GetGeoTransform()
    info["Projection"] = raster.GetProjection()
    nd = band.GetNoDataValue()
    info["NoDataValue"] = nd if nd is not None else -32768

    del band
    del raster

    return info, arr


def WriteGeoTiff(info, arr, filename, gdal_dtype=None, compress=True):
    filename = str(filename)
    parent_dir = os.path.dirname(filename)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    if gdal_dtype is None:
        if np.issubdtype(arr.dtype, np.floating):
            gdal_dtype = gdal.GDT_Float32
        elif arr.dtype == np.int32 or arr.dtype == np.uint32:
            gdal_dtype = gdal.GDT_Int32
        else:
            gdal_dtype = gdal.GDT_Int16

    driver = gdal.GetDriverByName("GTiff")
    options = ["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"] if compress else []
    raster = driver.Create(
        filename,
        int(info["RasterXSize"]),
        int(info["RasterYSize"]),
        1,
        gdal_dtype,
        options=options,
    )

    raster.SetGeoTransform(info["GeoTransform"])
    raster.SetProjection(info["Projection"])

    nodata = info.get("NoDataValue", -32768)
    if nodata is not None:
        raster.GetRasterBand(1).SetNoDataValue(float(nodata))
    raster.GetRasterBand(1).WriteArray(arr)

    raster.FlushCache()

    del raster
