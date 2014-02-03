import java.io.BufferedReader;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.*;

import org.json.JSONException;
import org.json.JSONObject;

import org.ejml.simple.*;
import org.ejml.data.*;
import org.ejml.ops.*;


public class FeatureFactory {
	HashMap<String, Integer> wordToNum = new HashMap<String, Integer>(); 
	HashMap<Integer, String> numToWord = new HashMap<Integer, String>();
	SimpleMatrix allVecs;
	
	
	
	
	/** Add any necessary initialization steps for your features here.
	 *  Using this constructor is optional. Depending on your
	 *  features, you may not need to intialize anything.
	 */
	public FeatureFactory() {

	}




	/** Do not modify this method **/
	public List<Datum> readData(String filename) throws IOException {

		List<Datum> data = new ArrayList<Datum>();
		BufferedReader in = new BufferedReader(new FileReader(filename));

		for (String line = in.readLine(); line != null; line = in.readLine()) {
			if (line.trim().length() == 0) {
				continue;
			}
			String[] bits = line.split("\\s+");
			String word = bits[0];
			String label = bits[1];

			Datum datum = new Datum(word, label);
			data.add(datum);
		}

		return data;
	}





	/** Do not modify this method **/
	public void readWordVectors(String vecFilename,String vocabFilename) throws IOException {
		// could be a parameter 
		int vectorSize = 50;
		
		// reading in vocab list
		BufferedReader in = new BufferedReader(new FileReader(vocabFilename));
		int counter = 0;
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			String[] bits = line.split("\\s+");
			String word = bits[0];
			wordToNum.put(word, counter); 
			numToWord.put(counter,word);
			counter++;
		}
		
		// reading in matrix
		allVecs = new SimpleMatrix(50, counter);
		in = new BufferedReader(new FileReader(vecFilename));
		counter = 0;
		for (String line = in.readLine(); line != null; line = in.readLine()) {
			String[] bits = line.split("\\s+");
			for (int pos=0;pos<vectorSize;pos++){ 
				allVecs.set(pos, counter, Double.parseDouble(bits[pos]));		
			}
			counter++;
		}
		assert(counter == wordToNum.size());

	}










}
