package com.neuralnetwork;

import com.neuralnetwork.cnn.filter.SimpleConvolutionFilterTest;
import com.neuralnetwork.cnn.layer.ConvolutionLayerTest;
import com.neuralnetwork.cnn.layer.SamplingLayerTest;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;

@RunWith(Suite.class)
@Suite.SuiteClasses({
        SimpleConvolutionFilterTest.class,
        ConvolutionLayerTest.class,
        SamplingLayerTest.class
})
public class WorkingCnnTests
{

}
