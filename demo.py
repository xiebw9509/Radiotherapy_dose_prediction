# @paper202305
import itk

TransformType = itk.Euler3DTransform[itk.D]

transform1 = TransformType.New()
params = transform1.GetParameters()

params[0] = 0.2
params[1] = 0.1
params[2] = 0.5
params[3] = 10
params[4] = 10
params[5] = 10

center1 = transform1.GetCenter()
center1[0] = 20
center1[1] = 30
center1[2] = 50

print(transform1.GetCenter())
transform1.SetParameters(params)

transform2 = TransformType.New()
params2 = transform2.GetParameters()

params2[0] = 0.3
params2[1] = 0.1
params2[2] = 0.1
params2[3] = 40
params2[4] = 40
params2[5] = 50

center2 = transform2.GetCenter()
center2[0] = 100
center2[1] = 200
center2[2] = 30

transform2.SetCenter(center2)
#print(transform2.GetCenter())
transform2.SetParameters(params2)

CompositeTransformType = itk.CompositeTransform[itk.D, 3]
outputCompositeTransform = CompositeTransformType.New()
outputCompositeTransform.AddTransform(transform1)
outputCompositeTransform.AddTransform(transform2)

invertTransform = outputCompositeTransform.GetInverseTransform()
#print(outputCompositeTransform)
print(invertTransform)
#print(invertTransform.GetParameters()[16])

VectorType = itk.Vector[itk.D, 3]
input = VectorType()
input[0] = 0
input[1] = 0
input[2] = 0
output = outputCompositeTransform.TransformPoint(input)
print(output)
output2 = invertTransform.TransformPoint(output)
print(output2)
