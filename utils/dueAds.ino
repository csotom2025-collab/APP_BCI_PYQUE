  /*

  Developed by Chip Audette (Fall 2013) for use with OpenBCI
  Builds upon work by Joel Murphy and Conor Russomanno (Summer 2013)
  
  Modified January 2014.
  
  This example uses the ADS1299 Arduino Library, a software bridge between the ADS1299 TI chip and 
  Arduino. See http://www.ti.com/product/ads1299 for more information about the device and the README
  folder in the ADS1299 directory for more information about the library.
  
  */
  typedef long int int32;

  #define N_CHANNELS_PER_OPENBCI (8)  //number of channels on a single OpenBCI board

  //for using a single OpenBCI board
  #include <ADS1299Managerdue.h>  //for a single OpenBCI board
  ADS1299Managerdue ADSManager; //Uses SPI bus and pins to say data is ready.  Uses Pins 13,12,11,10,9,8,4
  //#define MAX_N_CHANNELS (N_CHANNELS_PER_OPENBCI)   //how many channels are available in hardw- are
  #define MAX_N_CHANNELS (2*N_CHANNELS_PER_OPENBCI)   //how many channels are available in hardware...use this for daisy-chained board
  int nActiveChannels = MAX_N_CHANNELS;   //how many active channels would I like?


  //other settings for OpenBCI
  byte gainCode = ADS_GAIN24;   //how much gain do I want
  byte inputType = ADSINPUT_NORMAL;   //here's the normal way to setup the channels
  //byte inputType = ADSINPUT_SHORTED;  //here's another way to setup the channels
  //  byte inputType = ADSINPUT_TESTSIG;  //here's a third way to setup the channels

  //other variables
  long sampleCounter = 0;      // used to time the tesing loop
  boolean is_running = false;    // this flag is set in serialEvent on reciept of prompt
  #define PIN_STARTBINARY (7)  //pull this pin to ground to start binary transfer
  //define PIN_STARTBINARY_OPENEEG (6)
  boolean startBecauseOfPin = false;
  boolean startBecauseOfSerial = false;

  //analog input
  #define PIN_ANALOGINPUT (A0)
  int analogVal = 0;

  #define OUTPUT_NOTHING (0)
  #define OUTPUT_TEXT (1)
  #define OUTPUT_TEXT_1CHAN (9)
  #define OUTPUT_BINARY (2)
  #define OUTPUT_BINARY_SYNTHETIC (3)
  #define OUTPUT_BINARY_4CHAN (4)
  #define OUTPUT_BINARY_OPENEEG (6)
  #define OUTPUT_BINARY_OPENEEG_SYNTHETIC (7)
  #define OUTPUT_BINARY_WITH_AUX (8)
  int outputType;

  //Design filters  (This BIQUAD class requires ~6K of program space!  Ouch.)
  //For frequency response of these filters: http://www.earlevel.com/main/2010/12/20/biquad-calculator/
  #include <Biquad_multiChan.h>   //modified from this source code:  http://www.earlevel.com/main/2012/11/26/biquad-c-source-code/
  #define SAMPLE_RATE_HZ (250.0)  //default setting for OpenBCI
  #define FILTER_Q (0.5)        //critically damped is 0.707 (Butterworth)
  #define FILTER_PEAK_GAIN_DB (0.0) //we don't want any gain in the passband
  #define HP_CUTOFF_HZ (0.5)  //set the desired cutoff for the highpass filter
  Biquad_multiChan stopDC_filter(MAX_N_CHANNELS,bq_type_highpass,HP_CUTOFF_HZ / SAMPLE_RATE_HZ, FILTER_Q, FILTER_PEAK_GAIN_DB); //one for each channel because the object maintains the filter states
  //Biquad_multiChan stopDC_filter(MAX_N_CHANNELS,bq_type_bandpass,10.0 / SAMPLE_RATE_HZ, 6.0, FILTER_PEAK_GAIN_DB); //one for each channel because the object maintains the filter states
  #define NOTCH_FREQ_HZ (60.0)
  #define NOTCH_Q (4.0)              //pretty sharp notch
  #define NOTCH_PEAK_GAIN_DB (0.0)  //doesn't matter for this filter type
  Biquad_multiChan notch_filter1(MAX_N_CHANNELS,bq_type_notch,NOTCH_FREQ_HZ / SAMPLE_RATE_HZ, NOTCH_Q, NOTCH_PEAK_GAIN_DB); //one for each channel because the object maintains the filter states
  Biquad_multiChan notch_filter2(MAX_N_CHANNELS,bq_type_notch,NOTCH_FREQ_HZ / SAMPLE_RATE_HZ, NOTCH_Q, NOTCH_PEAK_GAIN_DB); //one for each channel because the object maintains the filter states
  boolean useFilters = false;  //enable or disable as you'd like...turn off if you're daisy chaining!

  unsigned long testStartTime = 0;
  boolean isTestMode = false;
  void setup() {

  //detect which version of OpenBCI we're using (is Pin2 jumped to Pin3?)
    int OpenBCI_version = OPENBCI_V2;  //assume V2
    //pinMode(2,INPUT);  digitalWrite(2,HIGH); //activate pullup...for detecting which version of OpenBCI PCB
    //pinMode(3,OUTPUT); digitalWrite(3,LOW);  //act as a ground pin...for detecting which version of OpenBCI PCB
    //if (digitalRead(2) == LOW) OpenBCI_version = OPENBCI_V1; //check pins to see if there is a jumper.  if so, it is the older board
    boolean isDaisy = false; if (MAX_N_CHANNELS > 8) isDaisy = true;

    ADSManager.initialize(OpenBCI_version,isDaisy);  //must do this VERY early in the setup...preferably first

    // setup the serial link to the PC
    if (MAX_N_CHANNELS > 8) {
      SerialUSB.begin(330400*2);  //Need 115200 for 16-channels, only need 115200 for 8-channels but let's do 115200*2 for consistency
    } else {
      SerialUSB.begin(330400);
    }
    
    while (!Serial);
    delay(5000);
    SerialUSB.println(F("ADS1299-Arduino UNO - Stream Raw Data")); //read the string from Flash to save RAM
    SerialUSB.print(F("Configured as OpenBCI_Version code = "));Serial.print(OpenBCI_version);// Serial.print(F(", isDaisy = "));Serial.println(ADSManager.isDaisy);
    SerialUSB.print(F("Configured for "));SerialUSB.print(MAX_N_CHANNELS); SerialUSB.println(F(" Channels"));
    SerialUSB.flush();
    
    // setup the channels as desired on the ADS1299..set gain, input type, referece (SRB1), and patient bias signal
    for (int chan=1; chan <= nActiveChannels; chan++) {
      ADSManager.activateChannel(chan, gainCode, inputType);
    }
    
    //setup the lead-off detection parameters
    //ADSManager.configureLeadOffDetection(LOFF_MAG_6NA, LOFF_FREQ_31p2HZ);

    //print state of all registers
    ADSManager.printAllRegisters();SerialUSB.flush();

    // setup hardware to allow a jumper or button to start the digitaltransfer
  //  pinMode(PIN_STARTBINARY,INPUT); digitalWrite(PIN_STARTBINARY,HIGH); //activate pullup
    //pinMode(PIN_STARTBINARY_OPENEEG,INPUT); digitalWrite(PIN_STARTBINARY_OPENEEG,HIGH);  //activate pullup
    
    
    
    // tell the controlling program that we're ready to start!
    SerialUSB.println(F("Press '?' to query and print ADS1299 register settings again")); //read it straight from flash
    SerialUSB.println(F("Press 1-8 to disable EEG Channels, q-i to enable (all enabled by default)"));
    SerialUSB.println(F("Press 'f' to enable filters.  'g' to disable filters"));
    SerialUSB.println(F("Press 'x' (text) "));    
    ADSManager.setSRB1(false);
    
    
  } // end of setup

  boolean firstReport = true;
  unsigned long totalMicrosBusy = 0;  //use this to count time
  void loop(){

    
      if (SerialUSB.available()) {
        
      char inChar = (char)SerialUSB.read();
      switch (inChar)
      {
      case 's':
          stopRunning();
          startBecauseOfSerial = is_running;
          break;
      case 'x':
          
          toggleRunState(OUTPUT_TEXT);
          startBecauseOfSerial = is_running;
          //if (is_running) SerialUSB.println(F("Arduino: Starting text..."));
          break;
      case 'p':
        if (!is_running) {
          sampleCounter = 0; // Reiniciamos el contador para ver cuántas muestras llegan
          isTestMode = true;
          testStartTime = millis();
          startRunning(OUTPUT_TEXT);
          SerialUSB.println(F("Iniciando prueba de frecuencia por 1 segundo..."));
        }
        break;
      }
    //}
  }   

    if (is_running) {
      //is data ready?     
      if (isTestMode && (millis() - testStartTime >= 1000)) {
      stopRunning();
      isTestMode = false;
      SerialUSB.print(F("Prueba terminada. Muestras capturadas en 1 seg: "));
      SerialUSB.println(sampleCounter);
      SerialUSB.print(F("Frecuencia estimada: "));
      SerialUSB.print(sampleCounter);
      SerialUSB.println(F(" Hz"));
    } else {
  
        
      while(!(ADSManager.isDataAvailable())){            // watch the DRDY pin
        delayMicroseconds(1);
      }
        

      //unsigned long start_micros = micros();
      //get the data
      //analogVal = analogRead(PIN_ANALOGINPUT);   // get analog value
      ADSManager.updateChannelData();            // update the channelData array 
      sampleCounter++;                           // increment my sample counter
    
      //print the data


          ADSManager.printChannelDataAsText(MAX_N_CHANNELS,sampleCounter);  //print all channels, whether active or not
      //ADSManager.sendFastBinary24(MAX_N_CHANNELS, sampleCounter);
      
    
      
  //    totalMicrosBusy += (micros()-start_micros); //accumulate
  //    if (sampleCounter==250) totalMicrosBusy = 0;  //start from 250th sample
  //    if (sampleCounter==500) {
  //      stopRunning();
  //      Serial.println();
  //      Serial.print(F("Was busy for "));
  //      Serial.print(totalMicrosBusy);
  //      Serial.println(F(" microseconds across 250 samples"));
  //      Serial.print(F("Assuming a 250Hz Sample Rate, it was busy for "));
  //      unsigned long micros_per_250samples = 1000000UL;
  //      Serial.print(((float)totalMicrosBusy/(float)micros_per_250samples)*100.0);
  //      Serial.println(F("% of the available time"));
  //    }
        
    } //end else
    }// end if (is_running
  } // end of loop


  #define ACTIVATE_SHORTED (2)
  #define ACTIVATE (1)
  #define DEACTIVATE (0)
  //void serialEvent(){            // send an 'x' on the serial line to trigger ADStest()
    //while(SerialUSB.available()){      
      
  boolean toggleRunState(int OUT_TYPE)
  {
    if (is_running) {
      return stopRunning();
    } else {
      return startRunning(OUT_TYPE);
    }
  }

  boolean stopRunning(void) {
    ADSManager.stop();                    // stop the data acquisition
    is_running = false;
    return is_running;
  }

  boolean startRunning(int OUT_TYPE) {
      outputType = OUT_TYPE;
      SerialUSB.println(F("ANTES DE START..."));
      ADSManager.start();    //start the data acquisition
      is_running = true;
      return is_running;
  }
