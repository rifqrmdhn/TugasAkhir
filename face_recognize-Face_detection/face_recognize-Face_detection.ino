#include "esp_camera.h"
#include <WiFi.h>
#include <WebSocketsServer.h>
#include "fd_forward.h" //fd

#define PWDN_GPIO_NUM     32
#define RESET_GPIO_NUM    -1
#define XCLK_GPIO_NUM      0
#define SIOD_GPIO_NUM     26
#define SIOC_GPIO_NUM     27

#define Y9_GPIO_NUM       35
#define Y8_GPIO_NUM       34
#define Y7_GPIO_NUM       39
#define Y6_GPIO_NUM       36
#define Y5_GPIO_NUM       21
#define Y4_GPIO_NUM       19
#define Y3_GPIO_NUM       18
#define Y2_GPIO_NUM        5
#define VSYNC_GPIO_NUM    25
#define HREF_GPIO_NUM     23
#define PCLK_GPIO_NUM     22

/* GPIO LED */
#define LED_FLASH 4
#define LED_ONOF 33

WebSocketsServer webSocket = WebSocketsServer(81);
WiFiServer server(80);

WiFiClient client;
String serverName = "192.168.18.23";
String serverPath = "/recognize"; 
const int serverPort = 80;

uint8_t cam_num;
bool connected = false;

const int timerInterval = 30000;    // time between each HTTP POST image
unsigned long previousMillis = 0;

String index_html =   "<html>\n \
<head>\n \
<title> Face Recognize </title>\n \
<script src='http://code.jquery.com/jquery-1.9.1.min.js'></script>\n \
</head>\n \
<body>\n \
<img id='live' src=''>\n \
</body>\n \
</html>\n \
<script>\n \
jQuery(function($){\n \
if (!('WebSocket' in window)) {\n \
alert('Your browser does not support web sockets');\n \
}else{\n \
setup();\n \
}\n \
function setup(){\n \
var host = 'ws://server_ip:81';\n \
var socket = new WebSocket(host);\n \
socket.binaryType = 'arraybuffer';\n \
if(socket){\n \
socket.onopen = function(){\n \
}\n \
socket.onmessage = function(msg){\n \
var bytes = new Uint8Array(msg.data);\n \
var binary= '';\n \
var len = bytes.byteLength;\n \
for (var i = 0; i < len; i++) {\n \
binary += String.fromCharCode(bytes[i])\n \
}\n \
var img = document.getElementById('live');\n \
img.src = 'data:image/jpg;base64,'+window.btoa(binary);\n \
}\n \
socket.onclose = function(){\n \
showServerResponse('The connection has been closed.');\n \
}\n \
}\n \
}\n \
});\n \
</script>";

void configCamera(){
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = Y2_GPIO_NUM;
  config.pin_d1 = Y3_GPIO_NUM;
  config.pin_d2 = Y4_GPIO_NUM;
  config.pin_d3 = Y5_GPIO_NUM;
  config.pin_d4 = Y6_GPIO_NUM;
  config.pin_d5 = Y7_GPIO_NUM;
  config.pin_d6 = Y8_GPIO_NUM;
  config.pin_d7 = Y9_GPIO_NUM;
  config.pin_xclk = XCLK_GPIO_NUM;
  config.pin_pclk = PCLK_GPIO_NUM;
  config.pin_vsync = VSYNC_GPIO_NUM;
  config.pin_href = HREF_GPIO_NUM;
  config.pin_sscb_sda = SIOD_GPIO_NUM;
  config.pin_sscb_scl = SIOC_GPIO_NUM;
  config.pin_pwdn = PWDN_GPIO_NUM;
  config.pin_reset = RESET_GPIO_NUM;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  config.frame_size = FRAMESIZE_SVGA;
  config.jpeg_quality = 10;
  config.fb_count = 1;

  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x", err);
    return;
  }
  sensor_t * s = esp_camera_sensor_get();
  s->set_vflip(s, 1);
  s->set_brightness(s, 2);     // -2 to 2
  s->set_contrast(s, 2);       // -2 to 2
  s->set_saturation(s, 2);     // -2 to 2
  s->set_special_effect(s, 0); // 0 to 6 (0 - No Effect, 1 - Negative, 2 - Grayscale, 3 - Red Tint, 4 - Green Tint, 5 - Blue Tint, 6 - Sepia)
  s->set_whitebal(s, 1);       // 0 = disable , 1 = enable
  s->set_awb_gain(s, 1);       // 0 = disable , 1 = enable
  s->set_wb_mode(s, 0);        // 0 to 4 - if awb_gain enabled (0 - Auto, 1 - Sunny, 2 - Cloudy, 3 - Office, 4 - Home)
}

mtmn_config_t mtmn_config = {0}; //fd
int detections = 0; //fd

void liveCam(uint8_t num){
  //capture a frame
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
      Serial.println("Frame buffer could not be acquired");
      return;
  }
  //replace this with your own function
  webSocket.sendBIN(num, fb->buf, fb->len);

  //return the frame buffer back to be reused
  esp_camera_fb_return(fb);
}

void webSocketEvent(uint8_t num, WStype_t type, uint8_t * payload, size_t length) {

    switch(type) {
        case WStype_DISCONNECTED:
            Serial.printf("[%u] Disconnected!\n", num);
            break;
        case WStype_CONNECTED:
            cam_num = num;
            connected = true;
            break;
        case WStype_TEXT:
        case WStype_BIN:
        case WStype_ERROR:      
        case WStype_FRAGMENT_TEXT_START:
        case WStype_FRAGMENT_BIN_START:
        case WStype_FRAGMENT:
        case WStype_FRAGMENT_FIN:
            break;
    }
}

void setup() {
  pinMode(LED_FLASH, OUTPUT);
  pinMode(LED_ONOF, OUTPUT);
  digitalWrite(LED_ONOF, LOW);
  digitalWrite(LED_FLASH, LOW);
  Serial.begin(115200);
  WiFi.begin("HaloDeck @2.4", "cobananyabumarn1");
  Serial.println("");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println("");
  String IP = WiFi.localIP().toString();
  Serial.print("IP address: " + IP);
  index_html.replace("server_ip", IP);
  server.begin();
  webSocket.begin();
  webSocket.onEvent(webSocketEvent);
  configCamera();

  mtmn_config = mtmn_init_config(); //fd
}
    
void http_resp(){
  WiFiClient client = server.available();
  if (client.connected() && client.available()) {                   
    client.flush();          
    client.print(index_html);
    client.stop();
  }
}

bool detected = false;
void loop() {
  http_resp();
  webSocket.loop();
  if(connected == true){
    liveCam(cam_num);
  }

  unsigned long currentMillis = millis();
  if (currentMillis - previousMillis >= timerInterval) {
    previousMillis = currentMillis;
    digitalWrite(LED_FLASH, HIGH);
    delay(1000);
  
    //face detection
    camera_fb_t * frame = NULL;
    frame = esp_camera_fb_get();
    
    dl_matrix3du_t *image_matrix = dl_matrix3du_alloc(1, frame->width, frame->height, 3);
    
    fmt2rgb888(frame->buf, frame->len, frame->format, image_matrix->item);
  
    esp_camera_fb_return(frame);
  
    box_array_t *boxes = face_detect(image_matrix, &mtmn_config);
  
    if (boxes != NULL) {
      detections = detections+1;
      Serial.printf("Faces detected %d times \n", detections);
      if (detections >= 5) {
        detections = 0;
      }
  
      dl_lib_free(boxes->score);
      dl_lib_free(boxes->box);
      dl_lib_free(boxes->landmark);
      dl_lib_free(boxes);
      sendPhoto();
    }
    
    dl_matrix3du_free(image_matrix);
    digitalWrite(LED_FLASH, LOW);
  }
}
String sendPhoto() {
  String getAll;
  String getBody;
  //digitalWrite(LED_FLASH, HIGH);
  //delay(3000);

  camera_fb_t * fb = NULL;
  fb = esp_camera_fb_get();
  digitalWrite(LED_FLASH, LOW);
  if(!fb) {
    Serial.println("Camera capture failed");
    delay(1000);
    ESP.restart();
  }
  
  Serial.println("Connecting to server: " + serverName);

  if (client.connect(serverName.c_str(), serverPort)) {
    Serial.println("Connection successful!");    
    String head = "--RandomNerdTutorials\r\nContent-Disposition: form-data; name=\"image\"; filename=\"esp321cam.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n";
    String tail = "\r\n--RandomNerdTutorials--\r\n";

    uint32_t imageLen = fb->len;
    uint32_t extraLen = head.length() + tail.length();
    uint32_t totalLen = imageLen + extraLen;
  
    client.println("POST " + serverPath + " HTTP/1.1");
    client.println("Host: " + serverName);
    client.println("Content-Length: " + String(totalLen));
    client.println("Content-Type: multipart/form-data; boundary=RandomNerdTutorials");
    client.println();
    client.print(head);
  
    uint8_t *fbBuf = fb->buf;
    size_t fbLen = fb->len;
    for (size_t n=0; n<fbLen; n=n+1024) {
      if (n+1024 < fbLen) {
        client.write(fbBuf, 1024);
        fbBuf += 1024;
      }
      else if (fbLen%1024>0) {
        size_t remainder = fbLen%1024;
        client.write(fbBuf, remainder);
      }
    }   
    client.print(tail);
    
    esp_camera_fb_return(fb);
    
    int timoutTimer = 10000;
    long startTimer = millis();
    boolean state = false;
    
    while ((startTimer + timoutTimer) > millis()) {
      Serial.print(".");
      delay(100);      
      while (client.available()) {
        char c = client.read();
        if (c == '\n') {
          if (getAll.length()==0) { state=true; }
          getAll = "";
        }
        else if (c != '\r') { getAll += String(c); }
        if (state==true) { getBody += String(c); }
        startTimer = millis();
      }
      if (getBody.length()>0) { break; }
    }
    Serial.println();
    client.stop();
    Serial.println(getBody);  
  }
  else {
    getBody = "Connection to " + serverName +  " failed.";
    Serial.println(getBody);
  }
  delay(1000);
  //digitalWrite(LED_FLASH, LOW);
  return getBody;
}