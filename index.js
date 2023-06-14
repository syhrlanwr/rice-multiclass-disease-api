const express = require('express');
const app = express();
const port = 3000;
const fileUpload = require('express-fileupload');
const tf = require('@tensorflow/tfjs-node');
const path = require('path');
const cors = require('cors');
const fs = require('fs');

app.use(cors());

app.use(fileUpload());

app.use(express.urlencoded({ extended: false }));
app.use(express.json());

app.use('/model', express.static(path.join(__dirname, 'model')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'index.html'));
});

app.post('/predict', async (req, res, next) => {
    try{
        if (!req.files) {
            return res.status(500).send({ msg: "file is not found" })
        }
        file = req.files.file;
        file.mv(path.join(__dirname, 'uploads', file.name), async (err) => {
            if (err) {
                return res.status(500).send({ msg: "Error occured" });
            }
            
            const img_path = path.join(__dirname, 'uploads', file.name);
            const img = fs.readFileSync(img_path);
            const decoded = tf.node.decodeImage(img, 3);
            const resized = tf.image.resizeBilinear(decoded, [224, 224]);
            const reshaped = resized.reshape([1, 224, 224, 3]);
            const normalized = reshaped.div(255.0);
            const model = await tf.loadGraphModel('http://localhost:3000/model/model.json');
            const prediction = model.predict(normalized);
            const predictionArray = prediction.arraySync();
            const labels = ['BrownSpot', 'Healthy', 'Hispa', 'LeafBlast'];
            const result = labels[predictionArray[0].indexOf(Math.max(...predictionArray[0]))];

            fs.unlinkSync(img_path);
            tf.disposeVariables();
            tf.dispose(model);
            tf.dispose(prediction);
            tf.dispose(predictionArray);
            tf.dispose(decoded);
            tf.dispose(resized);
            tf.dispose(reshaped);
            tf.dispose(normalized);
            
            res.send({ result: result, accuracy: Math.max(...predictionArray[0]).toFixed(2)*100 });
        });
    }
    catch(err){
        res.status(500).send({ msg: err });
    }
});


app.listen(port, () => {
    console.log(`Server running at http://localhost:${port}`);
});