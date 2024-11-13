https://drive.google.com/drive/folders/1k8jiRd0fI4oOcvs0Ox5Qaj5y5q7rXdvf

import java.io.FileInputStream;
import java.io.IOException;
import java.util.*;
import java.util.zip.CRC32;

public class CRC{
    public static long crc(String filePath) throws IOException
    {
        FileInputStream fileInputStream = new FileInputStream(filePath);
        CRC32 crc32 = new java.util.zip.CRC32();
        int byteRead;

        while((byteRead = fileInputStream.read()) != -1)
        {
            crc32.update(byteRead);
        }
        fileInputStream.close();
        return crc32.getValue();
    }
    public static void main(String[] args) throws IOException{
        String filePath = "C:\\Users\\Muhammed Arsath\\Downloads\\computer network_2 (3).jpg";
        long crcValue = crc(filePath);
        String binaryCrcValue = Long.toBinaryString(crcValue);
        System.out.println(binaryCrcValue);
    }
}
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;

public  class Checksum{

    public static Long checkSum(String filepath) throws IOException
    {

        FileInputStream fileInputStream = new FileInputStream(filepath);
        Long CheckSum = 0L;
        int byteRead ;

        while((byteRead = fileInputStream.read()) != -1)
        {
            CheckSum += byteRead;
        }

        fileInputStream.close();
        return CheckSum;


    }
    public static void main(String[] args) throws  IOException {

        String filePath = "C:\\Users\\Muhammed Arsath\\Downloads\\computer network_2 (3).jpg";
        Long checkSumValue = checkSum(filePath);
        String binaryValue = Long.toBinaryString(checkSumValue);
        System.out.println(binaryValue);
    }
}
