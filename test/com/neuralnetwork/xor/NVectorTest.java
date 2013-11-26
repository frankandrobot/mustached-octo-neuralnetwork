package com.neuralnetwork.xor;

import org.hamcrest.CoreMatchers;
import org.junit.Assert;
import org.junit.Test;

import static org.hamcrest.CoreMatchers.*;
import static org.junit.Assert.*;

public class NVectorTest
{
    @Test
    public void testConstructors() throws Exception
    {
        NVector a;
        //NVector(int size)
        a = new NVector(10);
        assertThat(a.size(), is(10));

        //NVector(double... aCoords)
        a = new NVector((double)10);
        assertThat(a.size(), is(1));
        assertThat(a.first(), is(10.0));

        //NVector(NVector vector)
        a = new NVector(new NVector(1,2,3));
        assertThat(a.size(), is(3));
        assertThat(a.get(0), is(1.0));
        assertThat(a.get(1), is(2.0));
        assertThat(a.get(2), is(3.0));

        //NVector(NVector vector, double... aCoords)
        a = new NVector(new NVector(1,2,3), -1, -2, -3);
        assertThat(a.size(), is(6));
        assertThat(a.get(0), is(1.0));
        assertThat(a.get(1), is(2.0));
        assertThat(a.get(2), is(3.0));
        assertThat(a.get(3), is(-1.0));
        assertThat(a.get(4), is(-2.0));
        assertThat(a.get(5), is(-3.0));
    }

    @Test
    public void testDot() throws Exception
    {
        NVector a = new NVector(1, 2, 3);
        NVector b = new NVector(-3, -2, -1);

        assertThat(a.dot(b), is(b.dot(a)));
        assertThat(a.dot(b), is(-3.0-4.0-3.0));
    }

    @Test
    public void testSize() throws Exception
    {
        NVector a = new NVector(1,2,3,4,5,6,7,8,9);
        assertThat(a.size(), is(9));
    }

    @Test
    public void testFirst() throws Exception
    {
        NVector a = new NVector(1,2,3,4,5,6,7,8,9);
        assertThat(a.first(), is(1.0));
    }

    @Test
    public void testLast() throws Exception
    {
        NVector a = new NVector(1,2,3,4,5,6,7,8,9);
        assertThat(a.last(), is(9.0));
    }

    @Test
    public void testSet() throws Exception
    {
        NVector a = new NVector(1,2,3,4,5,6,7,8,9);
        a.set(8,-9);
        assertThat(a.last(), is(-9.0));
    }

    @Test
    public void testGet() throws Exception
    {
        NVector b = new NVector(1,2,3,4,5,6,7,8,9);
        for(int i=0; i<9; i++)
            assertThat(b.get(i), is((double)i+1));
    }

    @Test
    public void testSubtract() throws Exception
    {
        NVector a = new NVector(-1,-2,-3);
        NVector b = new NVector(3,2,1);
        NVector c = a.subtract(b);

        System.out.println(a);
        assertThat(a.toString(), is("[-1.00000  -2.00000  -3.00000]"));
        assertThat(b.toString(), is("[3.00000  2.00000  1.00000]"));

        assertThat(c.toString(), is("[-4.00000  -4.00000  -4.00000]"));
    }

    @Test
    public void testError() throws Exception
    {
        NVector a = new NVector(1,2,3,4);
        assertThat(a.error(), is(1+4+9+16.0));
    }

    @Test
    public void testMylen() throws Exception
    {
        NVector a = new NVector(1,2,3,4);
        assertThat(a.sumOfCoords(), is(1+2+3+4.0));
    }

    @Test
    public void testConcatenate() throws Exception
    {
        NVector a = new NVector(0,1,2,3);
        NVector b = new NVector(4,5,6,7);
        NVector c = a.concatenate(b);

        assertThat(a.size(), is(4));
        assertThat(b.size(), is(4));
        for(int i=0; i<4; ++i)
        {
            assertThat(a.get(i), is((double)i));
            assertThat(b.get(i), is((double)i+4));
        }
        assertThat(c.size(), is(8));
        for(int i=0; i<c.size(); ++i)
        {
            assertThat(c.get(i), is((double)i));
        }
    }

    @Test
    public void testIterator() throws Exception
    {
        NVector a = new NVector(1,2,3,4,5,6,7,8,9);
        int len = 0;
        for(double coord:a)
            assertThat(coord, is((double)++len));
    }
}
