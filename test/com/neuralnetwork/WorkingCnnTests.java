package com.neuralnetwork;

import com.neuralnetwork.cnn.filter.SimpleConvolutionFilterTest;
import com.neuralnetwork.cnn.layer.ConvolutionMapTest;
import com.neuralnetwork.cnn.layer.SamplingMapTest;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({
        SimpleConvolutionFilterTest.class,
        ConvolutionMapTest.class,
        SamplingMapTest.class
})
public class WorkingCnnTests
{

}
