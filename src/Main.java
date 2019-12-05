import gpemlc.utils.KLabelsetGenerator;

public class Main {
	public static void main(String[] args) {
		KLabelsetGenerator klabelsetGen = new KLabelsetGenerator(3, 8, 12);
		klabelsetGen.generateKLabelsets();
	}
}
