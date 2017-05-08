package com.rieh.solution;

import java.awt.List;
import java.util.ArrayList;

public class Main {
	/* 1.1 Implement an algorithm to determine if a string has all unique characters. 
	 * (What if you cannot use additional data structure)
	 */
	
	public static boolean solve(String input) {
		char[] array = input.toCharArray();
		ArrayList<Character> seen = new ArrayList<Character>();
		for (char i : array) {
			if (seen.contains(i)) {
				System.out.println("false");
				return false;
			}
			seen.add(i);
		}
		System.out.println("true");
		return true;
	}
	
	public static void main(String[] args) {
		solve("abcifjrmen"); // true
		solve("abcifjcmek"); // false
	}
}
