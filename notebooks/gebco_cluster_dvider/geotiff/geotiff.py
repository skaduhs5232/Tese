from osgeo import gdal

def ReadGeoTiff(filename):
    raster = gdal.Open(filename)

    band = raster.GetRasterBand(1)
    arr = band.ReadAsArray()

    info = {}
    info['RasterYSize'] = raster.RasterYSize
    info['RasterXSize'] = raster.RasterXSize
    info['GeoTransform'] = raster.GetGeoTransform()
    info['Projection'] = raster.GetProjection()
    info['NoDataValue'] = band.GetNoDataValue()
    
    del band
    del raster

    return info, arr

def WriteGeoTiff(info, arr, filename):
    driver = gdal.GetDriverByName('GTiff')
    raster = driver.Create(filename,
                           info['RasterXSize'],
                           info['RasterYSize'],
                           1,
                           gdal.GDT_Int16)
    
    raster.SetGeoTransform(info['GeoTransform'])
    raster.SetProjection(info['Projection'])
    
    raster.GetRasterBand(1).SetNoDataValue(info['NoDataValue'])
    raster.GetRasterBand(1).WriteArray(arr)
    
    raster.FlushCache()
    
    del raster