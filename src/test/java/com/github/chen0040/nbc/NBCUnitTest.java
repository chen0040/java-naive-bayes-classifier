package com.github.chen0040.nbc;

import javax.xml.parsers.DocumentBuilderFactory;
import javax.xml.parsers.DocumentBuilder;
import javax.xml.parsers.ParserConfigurationException;

import org.w3c.dom.*;
import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.DataRow;
import org.testng.annotations.Test;
import org.xml.sax.SAXException;

import java.io.*;

import static org.testng.Assert.*;


/**
 * Created by xschen on 28/5/2017.
 */
public class NBCUnitTest {

   @Test
   public void TestHeartScale() throws FileNotFoundException {
      InputStream inputStream = NBCUnitTest.class.getClassLoader().getResourceAsStream("heart_scale");

      DataFrame dataFrame = DataQuery.libsvm().from(inputStream).build();


      dataFrame.unlock();
      for(int i=0; i < dataFrame.rowCount(); ++i){
         DataRow row = dataFrame.row(i);
         row.setCategoricalTargetCell("category-label", "" + row.target());
      }
      dataFrame.lock();

      NBC svc = new NBC();
      svc.fit(dataFrame);

      for(int i = 0; i < dataFrame.rowCount(); ++i){
         DataRow row = dataFrame.row(i);
         String predicted_label = svc.classify(row);
         System.out.println("predicted: "+predicted_label+"\texpected: "+row.categoricalTarget());
      }
   }

   @Test
   public void test_simple() throws ParserConfigurationException, IOException, SAXException {
      InputStream inputStream = NBCUnitTest.class.getClassLoader().getResourceAsStream("database.xml");
      DataFrame dataFrame = DataQuery.blank()
              .newInput("outlook")
              .newInput("temperature")
              .newInput("windy")
              .newInput("humidity")
              .newOutput("class")
              .end()
              .build();


      DocumentBuilderFactory dbFactory = DocumentBuilderFactory.newInstance();
      DocumentBuilder dBuilder = dbFactory.newDocumentBuilder();
      Document doc = dBuilder.parse(inputStream);

      doc.getDocumentElement().normalize();

      System.out.println("Root element :" + doc.getDocumentElement().getNodeName());

      NodeList nList = doc.getElementsByTagName("record");

      for(int i=0; i < nList.getLength(); ++i){
         Element node = (Element)nList.item(i);
         String outlook = node.getAttribute("outlook");
         String temperature = node.getAttribute("temperature");
         String humidity = node.getAttribute("humidity");
         String windy = node.getAttribute("windy");
         String output = node.getAttribute("class");

         DataRow row = dataFrame.newRow();

         row.setCategoricalCell("outlook", outlook);
         row.setCategoricalCell("windy", windy);
         row.setCell("temperature", Integer.parseInt(temperature));
         row.setCell("humidity", Integer.parseInt(humidity));
         row.setCategoricalTargetCell("class", output);

         dataFrame.addRow(row);
      }

      dataFrame.lock();

      NBC svc = new NBC();
      svc.fit(dataFrame);

      for(int i = 0; i < dataFrame.rowCount(); ++i){
         DataRow row = dataFrame.row(i);
         String predicted_label = svc.classify(row);
         System.out.println("predicted: "+predicted_label+"\texpected: "+row.categoricalTarget());
      }
   }
}
