// sample_bad_js.js —— 有问题的 JavaScript 代码（测试/演示用）
// 包含：eval()、无输入校验、全局变量污染、var 滥用

var globalData = [];
var config = { apiKey: "sk-js-123456", endpoint: "http://localhost" };

function processInput(userInput) {
    // eval 执行用户输入 —— 严重安全漏洞
    var result = eval(userInput);

    // 无输入校验
    globalData.push(result);

    // 全局变量污染
    for (var i = 0; i < globalData.length; i++) {
        for (var j = 0; j < globalData.length; j++) {
            console.log(globalData[i] + globalData[j]);
        }
    }

    return result;
}

function fetchData(url) {
    var xhr = new XMLHttpRequest();
    xhr.open("GET", url, false); // 同步请求阻塞
    xhr.send();
    return eval("(" + xhr.responseText + ")");
}

var handler = function (data) {
    // 无 try-catch
    var parsed = JSON.parse(data);
    globalData = parsed;
    processInput(parsed.command);
};
