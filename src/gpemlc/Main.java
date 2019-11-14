package gpemlc;

import java.util.Arrays;

public class Main {
	public static void main(String[] args) {
		String s = "(10 ((1 2 5) 10 9) (1 3 3))";
		
		Utils utils = new Utils();
		utils.reduce(s);
	}
}
