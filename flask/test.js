// test.js
const axios = require('axios');
const fs = require('fs');
const FormData = require('form-data');

const serverUrl = 'http://0.0.0.0:10005/predict';

async function testImagePrediction(imagePath) {
    const formData = new FormData();
    formData.append('file', fs.createReadStream(imagePath));

    try {
        const response = await axios.post(serverUrl, formData, {
            headers: formData.getHeaders()
        });
        console.log('Image Prediction Result:', response.data);
    } catch (error) {
        console.error('Error:', error.response ? error.response.data : error.message);
    }
}

// 테스트 실행
// testImagePrediction('./images/나.jpg');
testImagePrediction('./images/test.mp4');