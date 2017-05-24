package com.rieh.solution;

public class Main {
	/* 1.2 Implement an algorithm to reverse a string. 
	 */
	
	public static String reverseString(String input) {
		char[] array = input.toCharArray();
		int i = 0;
		char tempChar;
		while (i < (array.length)/2) {
			tempChar = array[i];
			array[i] = array[array.length-1-i];
			array[array.length-1-i] = tempChar;
			i++;
		}
		return new String(array);
	}
	
	public static void main(String[] args) {
		System.out.println(reverseString("not a man"));
		System.out.println(reverseString("ppapmyass"));
	}
}
