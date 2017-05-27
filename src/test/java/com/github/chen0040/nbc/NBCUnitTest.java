package com.github.chen0040.nbc;


import com.github.chen0040.data.frame.DataFrame;
import com.github.chen0040.data.frame.DataQuery;
import com.github.chen0040.data.frame.DataRow;
import org.testng.annotations.Test;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.InputStream;

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
}
