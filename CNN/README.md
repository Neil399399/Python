# CNN

## TFRecord
### TFRecord_Reader ( FileName, Image_Height, Image_Width, Image_Depth )
1. FileName : .TFRecord file.
2. Image_Height : 640 (defult)
3. Image_Width : 640 (defult)
4. Image_Depth : 3 (defult R,G,B)
### TFRecord _Writer ( Images, Labels, Images_Dir, TFrecord_name, Batch_Size )
1. Image_Dir : your folder path.
2. Image_Folder_List : sub folder name.

## CNN
### CNN_Model ( Image, Image_height, Image_width, Conv1_Filter, Conv2_Filter, Pool_Size, Padding, Activation_Function, output_layer )
Ex : CNN_Model ( image,640,640,6,36,2,'same',tf.nn.relu,2 )
 
* The `one_hot_depth` should equal output_layer units.
* Used `np.argmax()` to validate the value that predict by model.

## Note

* 需要將訓練資料放在 `train` 資料夾內(不能有任何子資料夾，否則會抓不到該檔案)。
* 需要將測試資料分類好並放在 `example_data` 內，本程式會以`example_data` 內的資料為基礎產生 file(image)_list.
* 產出的 `.tfrecord` 檔案會放置在 `TFRecord`。
* 完成以上步驟後請先執行 `TFRecord_maker.py`。