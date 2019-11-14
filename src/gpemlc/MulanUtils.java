package gpemlc;

import mulan.data.InvalidDataFormatException;
import mulan.data.MultiLabelInstances;
import net.sf.jclec.util.random.IRandGen;
import weka.core.Instances;

public class MulanUtils {

	/**
	 * Partition data into train and validation sets
	 * 
	 * @param mlData Full data
	 * @param samplingTechnique Technique for selecting the data
	 * @return Sampled dataset
	 */
	public static MultiLabelInstances sampleData(MultiLabelInstances mlData, double ratio, IRandGen randgen){
		MultiLabelInstances newMLData = null;
		Instances data, newData;
		
		//Create new empty dataset
		data = mlData.getDataSet();
		newData = new Instances(mlData.getDataSet());
		newData.removeAll(newData);
		
		//Get shuffle array of indexes
		int [] indexes = new int[mlData.getNumInstances()];
		for(int i=0; i<mlData.getNumInstances(); i++) {
			indexes[i] = i;
		}
		int r, aux;
		for(int i=0; i<mlData.getNumInstances(); i++) {
			r = randgen.choose(mlData.getNumInstances());
			aux = indexes[i];
			indexes[i] = indexes[r];
			indexes[r] = aux;
		}
		
		//Number of instances to keep
		int limit = (int)Math.round(mlData.getNumInstances() * ratio); 

		//Add corresponding random instances to new data
		for(int i=0; i<limit; i++) {
			newData.add(data.get(indexes[i]));
		}
		
		data = null;
		
		//Generate new multi-label dataset
		try {
			newMLData = new MultiLabelInstances(newData, mlData.getLabelsMetaData());
		} catch (InvalidDataFormatException e1) {
			e1.printStackTrace();
		}
		
		return newMLData;
	}
}
