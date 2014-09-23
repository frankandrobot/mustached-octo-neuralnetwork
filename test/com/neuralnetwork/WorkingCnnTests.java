package com.neuralnetwork;

import com.neuralnetwork.cnn.CnnTest;
import com.neuralnetwork.cnn.filter.SimpleConvolutionFilterTest;
import com.neuralnetwork.cnn.map.ConvolutionMapTest;
import com.neuralnetwork.cnn.map.SamplingMapTest;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({
        SimpleConvolutionFilterTest.class,
        ConvolutionMapTest.class,
        SamplingMapTest.class,
        CnnTest.class
})
public class WorkingCnnTests
{

}
