import odl

geometry = odl.tomo.ConeFlatGeometry()
operator = odl.tomo.RayTransform(space, geometry)

