# Installation of Spark
Download Spark from http://spark.apache.org/downloads.html 

Download Scala from http://www.scala-lang.org/download (find the version number in README inside spark archive). 

Say the unpacked Spark and Scala folders are located at:

```
PARK_HOME=/usr/local/bin/spark
SCALA_Home=/usr/local/bin/scala
```

Go to Spark root directory (SPARK_HOME) and run in command line: sbt/sbt clean assembly

Then start up Spark, also from Spark root folder: ./bin/spark-shell

The following need to be added to .bashrc:

```
export SCALA_Home=/usr/local/bin/scala
export SPARK_HOME=/usr/local/bin/spark
export PATH = $PATH:$SCALA_HOME/bin:$SPARK_HOME:$SPARK_HOME/bin
export PYTHONPATH=$PYTHONPATH:$SPARK_HOME/python:$SPARK_HOME/python/lib/py4j-0.8.2.1-src.zip
```

**If Macports is used to install Python, the following might to be in the path for thunder-python (or other programs installed by pip) to be found**

```
export PATH = $PATH:/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin
```

# Installation of Thunder
github.com/thunder-project/thunder

For me I had to find the installed thunder executables (thunder, thunder-submit) and add their path to PATH in .bashrc:

```
/opt/local/Library/Frameworks/Python.framework/Versions/2.7/bin
```