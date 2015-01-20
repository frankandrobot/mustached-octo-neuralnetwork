package com.neuralnetwork.helpers;

import static org.hamcrest.CoreMatchers.is;
import static org.junit.Assert.assertThat;

public class NumberAssert {

    public static String toStr(double x)
    {
        return toStr(6,x);
    }

    public static String toStr(int precision, double x)
    {
        return String.format("%."+precision+"f", x);
    }

    public static void _assert(double expected, double actual)
    {
        assertThat(toStr(expected), is(toStr(actual)));
    }

    public static void _assert(String reason, double expected, double actual)
    {
        assertThat(reason, toStr(expected), is(toStr(actual)));
    }

    public static void _assert(double[] expected, double[] actual)
    {
        assertThat(expected.length, is(actual.length));

        for(int i=0; i<expected.length; ++i)
        {
            _assert(expected[i], actual[i]);
        }
    }

    public static void _assert(String reason, double[] expected, double[] actual)
    {
        assertThat(reason, expected.length, is(actual.length));

        for(int i=0; i<expected.length; ++i)
        {
            _assert(reason, expected[i], actual[i]);
        }
    }
}
