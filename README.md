# StrawberryBot

"fine, strawberries r like the weird cousins of the fruit world, all sweet n stuff but also kinda... seedy, dunno why u care tho, we were in the middle of somethin else"
-DumBott



### howto: setup training data and training ai

ok, so you want to teach a ai how to do your bidding? i gotchu
1. go to https://www.makesense.ai/ and label your images. (NOTE: website does NOT save your stuff). important the project csv to extend current csv
2. when done, export to csv and put it in the python folder, replace the old one if it exists
3. replace all the images in dataset/images with the images you used for the csv, file names must match with the ones you used on the website
4. run convert_csv.py, this fills the labels folder and makes the current data useable for training
5. run convert_aug.py, this creates variations of the images and makes the original ones the verification set. this increases images by x10!
6. finally, run AiTrain.py, this actually trains the ai. make sure to config the file and data.yaml with epochs and such
7. after a while, its done and the model will be in runs/detect/train-X, where X is the latest ran number. in /weights/best.pt is your model
8. use the model where you need it. in StrawCV theres a variable asking for this path, and the script will use that model
9. ur done. note that you'll only get different results based on labels and images provided. more epochs will only rarely show changes.

also be a dear and give used/good models a proper name, like basev5 (v5 of the basic trained model that functionally worked alright)