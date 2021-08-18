import pattern
import generator




checkBoard = pattern.Checker(12,3)

checkBoard.draw()
checkBoard.show()



circleBoard = pattern.Circle(500, 50, (200,100) )

circleBoard.draw()
circleBoard.show()


RGBSpecturm = pattern.Spectrum(100)
RGBSpecturm.draw()
RGBSpecturm.show()



label_path = 'Labels.json'
file_path = "exercise_data"

gen = generator.ImageGenerator(file_path, label_path, batch_size=9, image_size=(64,64), rotation=True, mirroring=False, shuffle=True)
gen.show()

