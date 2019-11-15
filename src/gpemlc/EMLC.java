package gpemlc;

import java.util.Hashtable;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import mulan.classifier.InvalidDataException;
import mulan.classifier.MultiLabelLearner;
import mulan.classifier.MultiLabelOutput;
import mulan.classifier.meta.MultiLabelMetaLearner;
import mulan.data.MultiLabelInstances;
import weka.core.Instance;
import weka.core.TechnicalInformation;

public class EMLC extends MultiLabelMetaLearner {

	/**
	 * 
	 */
	private static final long serialVersionUID = 3236560089372633198L;

	/**
	 * Table including classifiers
	 */
	Hashtable<String, MultiLabelLearner> table;
	
	Hashtable<String, Prediction> tablePredictions = new Hashtable<String, Prediction>();
	
	String genotype;
	
	public void setTable(Hashtable<String, MultiLabelLearner> table) {
		this.table = table;
	}
	
	public void setGenotype(String genotype) {
		this.genotype = genotype;
	}
	
	public EMLC(MultiLabelLearner baseLearner) {
		super(baseLearner);
		// TODO Auto-generated constructor stub
	}

	@Override
	protected void buildInternal(MultiLabelInstances trainingSet) throws Exception {
//		this.numLabels = trainingSet.getNumLabels();
		// TODO Auto-generated method stub

	}

	@Override
	protected MultiLabelOutput makePredictionInternal(Instance instance) throws Exception, InvalidDataException {
		byte [] pred = reduce(genotype, instance).bip[0];
		
		boolean[] bip = new boolean[numLabels];
		for(int i=0; i<numLabels; i++) {
			if(pred[i] > 0) {
				bip[i] = true;
			}
			else {
				bip[i] = false;
			}
		}
		
		return new MultiLabelOutput(bip);
	}
	
	public Prediction reduce(String ind, Instance instance) {
		Pattern pattern = Pattern.compile("\\((_?\\d+ )+_?\\d+\\)");
		
		Matcher m = pattern.matcher(ind);
		int count = 0;
		Prediction pred = new Prediction(1, numLabels);
		
		while(m.find()) {
			pred = combine(m.group(0), table, instance);
			tablePredictions.put("_"+count, pred);
			ind = ind.substring(0, m.start()) + "_" + count + ind.substring(m.end(), ind.length());
			count++;
			m = pattern.matcher(ind);
		}
		
//		System.out.println(Arrays.toString(pred.bip[0]));
		return pred;
	}
	
	protected Prediction combine(String s, Hashtable<String, MultiLabelLearner> table, Instance instance){
		Prediction pred = new Prediction(1, numLabels);
		
		Pattern pattern = Pattern.compile("\\d+");
		Matcher m;
		int n;
		int nPreds = 0;
//		System.out.println("s: " + s);
		
		String [] pieces = s.split(" ");
		for(String piece : pieces) {
//			System.out.println("piece: " + piece);
			m = pattern.matcher(piece);
			m.find();
			n = Integer.parseInt(m.group(0));
//			System.out.println("n: " + n);
			if(piece.contains("_")) {
				pred.addPrediction(tablePredictions.get("_"+n));
				table.remove("_"+n);
				nPreds++;
			}
			else {
//				pred.addPrediction(getPredictions(table.get(String.valueOf(n))));
				pred.addPrediction(getPredictions(table.get(String.valueOf(n)), instance));
				nPreds++;
			}
		}
		
//		System.out.println("sum: " + Arrays.toString(pred.bip[0]));
		pred.divideAndThresholdPrediction(nPreds, 0.5);
//		System.out.println("div: " + Arrays.toString(pred.bip[0]));
		
		return pred;
	}

	protected byte[][] getPredictions(MultiLabelLearner learner, Instance instance){
		byte[][] bip = new byte[1][numLabels];
		
		try {
			boolean[] boolPred = learner.makePrediction(instance).getBipartition();
			for(int j=0; j<numLabels; j++) {
				if(boolPred[j]) {
					bip[0][j] = 1;
				}
			}
		}catch(Exception e) {
			e.printStackTrace();
		}
		
		return bip;
	}

	@Override
	public TechnicalInformation getTechnicalInformation() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public String globalInfo() {
		// TODO Auto-generated method stub
		return null;
	}

}
