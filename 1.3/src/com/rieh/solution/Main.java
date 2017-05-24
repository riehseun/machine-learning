package com.rieh.solution;

import java.util.ArrayList;

public class Main {
	/* 1.3 Implement an algorithm to determine if one string 
	 * is a permutation of the other. 
	 */
	
	public static boolean isPermutation(String input1, String input2) {
		char[] array1 = input1.toCharArray();
		char[] array2 = input2.toCharArray();
		
		ArrayList<Character> list1 = new ArrayList<Character>();
		ArrayList<Character> list2 = new ArrayList<Character>();
		
		for (char i : array1) {
			list1.add(i);
		}
		for (char j : array2) {
			list2.add(j);
		}
		
		// For each char in list 2, if that character is found in list1, 
		// remove that character in list1
		for (Character i : list2) {
			if (list1.contains(i)) {
				list1.remove(list1.indexOf(i));
			}
		}
		
		if (list1.size() == 0) {
			return true;
		}
		else {
			return false;
		}
	}
	
	public static void main(String[] args) {
		System.out.println(isPermutation("notaman", "tanamno"));
		System.out.println(isPermutation("ppapmyass", "doppskas"));
		System.out.println(isPermutation("notaman", "tnmaona"));
		System.out.println(isPermutation("notaman", "tnmaont"));
	}
}
