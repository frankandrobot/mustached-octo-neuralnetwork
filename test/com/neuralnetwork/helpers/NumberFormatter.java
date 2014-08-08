package com.neuralnetwork.helpers;

public class NumberFormatter {

    public static String toStr(double x)
    {
        return toStr(6,x);
    }

    public static String toStr(int precision, double x)
    {
        return String.format("%."+precision+"f", x);
    }
}
