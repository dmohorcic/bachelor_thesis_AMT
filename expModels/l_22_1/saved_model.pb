©‘	
—µ
B
AssignVariableOp
resource
value"dtype"
dtypetypeИ
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
Ы
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
В
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(И

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
Ц
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.4.12unknown8нф
Д
conv2d_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_12/kernel
}
$conv2d_12/kernel/Read/ReadVariableOpReadVariableOpconv2d_12/kernel*&
_output_shapes
:*
dtype0
t
conv2d_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_12/bias
m
"conv2d_12/bias/Read/ReadVariableOpReadVariableOpconv2d_12/bias*
_output_shapes
:*
dtype0
Д
conv2d_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_13/kernel
}
$conv2d_13/kernel/Read/ReadVariableOpReadVariableOpconv2d_13/kernel*&
_output_shapes
: *
dtype0
t
conv2d_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_13/bias
m
"conv2d_13/bias/Read/ReadVariableOpReadVariableOpconv2d_13/bias*
_output_shapes
: *
dtype0
y
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А	X*
shared_namedense_6/kernel
r
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel*
_output_shapes
:	А	X*
dtype0
p
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*
shared_namedense_6/bias
i
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes
:X*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
t
true_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nametrue_positives
m
"true_positives/Read/ReadVariableOpReadVariableOptrue_positives*
_output_shapes
:*
dtype0
v
false_positivesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_positives
o
#false_positives/Read/ReadVariableOpReadVariableOpfalse_positives*
_output_shapes
:*
dtype0
x
true_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nametrue_positives_1
q
$true_positives_1/Read/ReadVariableOpReadVariableOptrue_positives_1*
_output_shapes
:*
dtype0
v
false_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:* 
shared_namefalse_negatives
o
#false_negatives/Read/ReadVariableOpReadVariableOpfalse_negatives*
_output_shapes
:*
dtype0
y
true_positives_2VarHandleOp*
_output_shapes
: *
dtype0*
shape:»*!
shared_nametrue_positives_2
r
$true_positives_2/Read/ReadVariableOpReadVariableOptrue_positives_2*
_output_shapes	
:»*
dtype0
u
true_negativesVarHandleOp*
_output_shapes
: *
dtype0*
shape:»*
shared_nametrue_negatives
n
"true_negatives/Read/ReadVariableOpReadVariableOptrue_negatives*
_output_shapes	
:»*
dtype0
{
false_positives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:»*"
shared_namefalse_positives_1
t
%false_positives_1/Read/ReadVariableOpReadVariableOpfalse_positives_1*
_output_shapes	
:»*
dtype0
{
false_negatives_1VarHandleOp*
_output_shapes
: *
dtype0*
shape:»*"
shared_namefalse_negatives_1
t
%false_negatives_1/Read/ReadVariableOpReadVariableOpfalse_negatives_1*
_output_shapes	
:»*
dtype0
Т
Adam/conv2d_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_12/kernel/m
Л
+Adam/conv2d_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/m*&
_output_shapes
:*
dtype0
В
Adam/conv2d_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_12/bias/m
{
)Adam/conv2d_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/m*
_output_shapes
:*
dtype0
Т
Adam/conv2d_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_13/kernel/m
Л
+Adam/conv2d_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/m*&
_output_shapes
: *
dtype0
В
Adam/conv2d_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_13/bias/m
{
)Adam/conv2d_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/m*
_output_shapes
: *
dtype0
З
Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А	X*&
shared_nameAdam/dense_6/kernel/m
А
)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m*
_output_shapes
:	А	X*
dtype0
~
Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*$
shared_nameAdam/dense_6/bias/m
w
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes
:X*
dtype0
Т
Adam/conv2d_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_12/kernel/v
Л
+Adam/conv2d_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/kernel/v*&
_output_shapes
:*
dtype0
В
Adam/conv2d_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_12/bias/v
{
)Adam/conv2d_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_12/bias/v*
_output_shapes
:*
dtype0
Т
Adam/conv2d_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv2d_13/kernel/v
Л
+Adam/conv2d_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/kernel/v*&
_output_shapes
: *
dtype0
В
Adam/conv2d_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *&
shared_nameAdam/conv2d_13/bias/v
{
)Adam/conv2d_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_13/bias/v*
_output_shapes
: *
dtype0
З
Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	А	X*&
shared_nameAdam/dense_6/kernel/v
А
)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v*
_output_shapes
:	А	X*
dtype0
~
Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:X*$
shared_nameAdam/dense_6/bias/v
w
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes
:X*
dtype0

NoOpNoOp
µ8
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*р7
valueж7Bг7 B№7
І
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
R
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
R
#regularization_losses
$trainable_variables
%	variables
&	keras_api
R
'regularization_losses
(trainable_variables
)	variables
*	keras_api
R
+regularization_losses
,trainable_variables
-	variables
.	keras_api
h

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
Ј
5iter

6beta_1

7beta_2
	8decay
9learning_ratemmАmБmВ/mГ0mДvЕvЖvЗvИ/vЙ0vК
 
*
0
1
2
3
/4
05
*
0
1
2
3
/4
05
≠

regularization_losses
:non_trainable_variables
;layer_regularization_losses
<metrics

=layers
>layer_metrics
trainable_variables
	variables
 
\Z
VARIABLE_VALUEconv2d_12/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_12/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
regularization_losses
?non_trainable_variables
@layer_regularization_losses
Ametrics

Blayers
Clayer_metrics
trainable_variables
	variables
 
 
 
≠
regularization_losses
Dnon_trainable_variables
Elayer_regularization_losses
Fmetrics

Glayers
Hlayer_metrics
trainable_variables
	variables
 
 
 
≠
regularization_losses
Inon_trainable_variables
Jlayer_regularization_losses
Kmetrics

Llayers
Mlayer_metrics
trainable_variables
	variables
\Z
VARIABLE_VALUEconv2d_13/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEconv2d_13/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
≠
regularization_losses
Nnon_trainable_variables
Olayer_regularization_losses
Pmetrics

Qlayers
Rlayer_metrics
 trainable_variables
!	variables
 
 
 
≠
#regularization_losses
Snon_trainable_variables
Tlayer_regularization_losses
Umetrics

Vlayers
Wlayer_metrics
$trainable_variables
%	variables
 
 
 
≠
'regularization_losses
Xnon_trainable_variables
Ylayer_regularization_losses
Zmetrics

[layers
\layer_metrics
(trainable_variables
)	variables
 
 
 
≠
+regularization_losses
]non_trainable_variables
^layer_regularization_losses
_metrics

`layers
alayer_metrics
,trainable_variables
-	variables
ZX
VARIABLE_VALUEdense_6/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_6/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

/0
01

/0
01
≠
1regularization_losses
bnon_trainable_variables
clayer_regularization_losses
dmetrics

elayers
flayer_metrics
2trainable_variables
3	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 

g0
h1
i2
j3
8
0
1
2
3
4
5
6
7
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ktotal
	lcount
m	variables
n	keras_api
W
o
thresholds
ptrue_positives
qfalse_positives
r	variables
s	keras_api
W
t
thresholds
utrue_positives
vfalse_negatives
w	variables
x	keras_api
p
ytrue_positives
ztrue_negatives
{false_positives
|false_negatives
}	variables
~	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

k0
l1

m	variables
 
a_
VARIABLE_VALUEtrue_positives=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_positives>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUE

p0
q1

r	variables
 
ca
VARIABLE_VALUEtrue_positives_1=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUE
ca
VARIABLE_VALUEfalse_negatives>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

u0
v1

w	variables
ca
VARIABLE_VALUEtrue_positives_2=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEtrue_negatives=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_positives_1>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUE
ec
VARIABLE_VALUEfalse_negatives_1>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUE

y0
z1
{2
|3

}	variables
}
VARIABLE_VALUEAdam/conv2d_12/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_12/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_13/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_13/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_12/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_12/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}
VARIABLE_VALUEAdam/conv2d_13/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
{y
VARIABLE_VALUEAdam/conv2d_13/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_6/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_6/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
Ф
serving_default_conv2d_12_inputPlaceholder*0
_output_shapes
:€€€€€€€€€ј*
dtype0*%
shape:€€€€€€€€€ј
≠
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_12_inputconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_6/kerneldense_6/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В */
f*R(
&__inference_signature_wrapper_21511446
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ѓ
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_12/kernel/Read/ReadVariableOp"conv2d_12/bias/Read/ReadVariableOp$conv2d_13/kernel/Read/ReadVariableOp"conv2d_13/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp"true_positives/Read/ReadVariableOp#false_positives/Read/ReadVariableOp$true_positives_1/Read/ReadVariableOp#false_negatives/Read/ReadVariableOp$true_positives_2/Read/ReadVariableOp"true_negatives/Read/ReadVariableOp%false_positives_1/Read/ReadVariableOp%false_negatives_1/Read/ReadVariableOp+Adam/conv2d_12/kernel/m/Read/ReadVariableOp)Adam/conv2d_12/bias/m/Read/ReadVariableOp+Adam/conv2d_13/kernel/m/Read/ReadVariableOp)Adam/conv2d_13/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp+Adam/conv2d_12/kernel/v/Read/ReadVariableOp)Adam/conv2d_12/bias/v/Read/ReadVariableOp+Adam/conv2d_13/kernel/v/Read/ReadVariableOp)Adam/conv2d_13/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOpConst*.
Tin'
%2#	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В **
f%R#
!__inference__traced_save_21511803
Х
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_12/kernelconv2d_12/biasconv2d_13/kernelconv2d_13/biasdense_6/kerneldense_6/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttrue_positivesfalse_positivestrue_positives_1false_negativestrue_positives_2true_negativesfalse_positives_1false_negatives_1Adam/conv2d_12/kernel/mAdam/conv2d_12/bias/mAdam/conv2d_13/kernel/mAdam/conv2d_13/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/conv2d_12/kernel/vAdam/conv2d_12/bias/vAdam/conv2d_13/kernel/vAdam/conv2d_13/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/v*-
Tin&
$2"*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *-
f(R&
$__inference__traced_restore_21511912Ве
л
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_21511256

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:€€€€€€€€€I :W S
/
_output_shapes
:€€€€€€€€€I 
 
_user_specified_nameinputs
…
f
-__inference_dropout_13_layer_call_fn_21511645

inputs
identityИҐStatefulPartitionedCallй
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_215112512
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€I 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€I 
 
_user_specified_nameinputs
љ
I
-__inference_dropout_13_layer_call_fn_21511650

inputs
identity—
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_215112562
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€I :W S
/
_output_shapes
:€€€€€€€€€I 
 
_user_specified_nameinputs
Ў

а
G__inference_conv2d_12_layer_call_and_return_conditional_losses_21511567

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€©*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€©2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€©2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ј::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
ф	
ё
E__inference_dense_6_layer_call_and_return_conditional_losses_21511672

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А	X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€X2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€X2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€X2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А	
 
_user_specified_nameinputs
К
Б
,__inference_conv2d_12_layer_call_fn_21511576

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallГ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_215111652
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€©2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ј::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
п
f
H__inference_dropout_12_layer_call_and_return_conditional_losses_21511198

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€©2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€©:X T
0
_output_shapes
:€€€€€€€€€©
 
_user_specified_nameinputs
“

а
G__inference_conv2d_13_layer_call_and_return_conditional_losses_21511614

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€I *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€I 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
њ
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_21511656

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А	2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€$ :W S
/
_output_shapes
:€€€€€€€€€$ 
 
_user_specified_nameinputs
Ќ
f
-__inference_dropout_12_layer_call_fn_21511598

inputs
identityИҐStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_215111932
StatefulPartitionedCallЧ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:€€€€€€€€€©2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€©22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€©
 
_user_specified_nameinputs
Д
j
N__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_21511132

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Ў

а
G__inference_conv2d_12_layer_call_and_return_conditional_losses_21511165

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:*
dtype02
Conv2D/ReadVariableOp•
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€©*
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOpЙ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€©2	
BiasAdda
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2
Relu†
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:€€€€€€€€€©2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:€€€€€€€€€ј::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
–:
њ
B__inference_m_22_layer_call_and_return_conditional_losses_21511491

inputs,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identityИҐ conv2d_12/BiasAdd/ReadVariableOpҐconv2d_12/Conv2D/ReadVariableOpҐ conv2d_13/BiasAdd/ReadVariableOpҐconv2d_13/Conv2D/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOp≥
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_12/Conv2D/ReadVariableOp√
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€©*
paddingVALID*
strides
2
conv2d_12/Conv2D™
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp±
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€©2
conv2d_12/BiasAdd
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2
conv2d_12/Reluy
dropout_12/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_12/dropout/Const≥
dropout_12/dropout/MulMulconv2d_12/Relu:activations:0!dropout_12/dropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2
dropout_12/dropout/MulА
dropout_12/dropout/ShapeShapeconv2d_12/Relu:activations:0*
T0*
_output_shapes
:2
dropout_12/dropout/Shapeё
/dropout_12/dropout/random_uniform/RandomUniformRandomUniform!dropout_12/dropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€©*
dtype021
/dropout_12/dropout/random_uniform/RandomUniformЛ
!dropout_12/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_12/dropout/GreaterEqual/yу
dropout_12/dropout/GreaterEqualGreaterEqual8dropout_12/dropout/random_uniform/RandomUniform:output:0*dropout_12/dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2!
dropout_12/dropout/GreaterEqual©
dropout_12/dropout/CastCast#dropout_12/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€©2
dropout_12/dropout/Castѓ
dropout_12/dropout/Mul_1Muldropout_12/dropout/Mul:z:0dropout_12/dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€©2
dropout_12/dropout/Mul_1 
max_pooling2d_12/MaxPoolMaxPooldropout_12/dropout/Mul_1:z:0*/
_output_shapes
:€€€€€€€€€T*
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPool≥
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_13/Conv2D/ReadVariableOpЁ
conv2d_13/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€I *
paddingVALID*
strides
2
conv2d_13/Conv2D™
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp∞
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
conv2d_13/Reluy
dropout_13/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout_13/dropout/Const≤
dropout_13/dropout/MulMulconv2d_13/Relu:activations:0!dropout_13/dropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
dropout_13/dropout/MulА
dropout_13/dropout/ShapeShapeconv2d_13/Relu:activations:0*
T0*
_output_shapes
:2
dropout_13/dropout/ShapeЁ
/dropout_13/dropout/random_uniform/RandomUniformRandomUniform!dropout_13/dropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€I *
dtype021
/dropout_13/dropout/random_uniform/RandomUniformЛ
!dropout_13/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2#
!dropout_13/dropout/GreaterEqual/yт
dropout_13/dropout/GreaterEqualGreaterEqual8dropout_13/dropout/random_uniform/RandomUniform:output:0*dropout_13/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2!
dropout_13/dropout/GreaterEqual®
dropout_13/dropout/CastCast#dropout_13/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€I 2
dropout_13/dropout/CastЃ
dropout_13/dropout/Mul_1Muldropout_13/dropout/Mul:z:0dropout_13/dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
dropout_13/dropout/Mul_1 
max_pooling2d_13/MaxPoolMaxPooldropout_13/dropout/Mul_1:z:0*/
_output_shapes
:€€€€€€€€€$ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPools
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
flatten_6/Const°
flatten_6/ReshapeReshape!max_pooling2d_13/MaxPool:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А	2
flatten_6/Reshape¶
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	А	X*
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMulflatten_6/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€X2
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€X2
dense_6/BiasAddy
dense_6/SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€X2
dense_6/Sigmoid≤
IdentityIdentitydense_6/Sigmoid:y:0!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
п
f
H__inference_dropout_12_layer_call_and_return_conditional_losses_21511593

inputs

identity_1c
IdentityIdentityinputs*
T0*0
_output_shapes
:€€€€€€€€€©2

Identityr

Identity_1IdentityIdentity:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2

Identity_1"!

identity_1Identity_1:output:0*/
_input_shapes
:€€€€€€€€€©:X T
0
_output_shapes
:€€€€€€€€€©
 
_user_specified_nameinputs
“

а
G__inference_conv2d_13_layer_call_and_return_conditional_losses_21511223

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐConv2D/ReadVariableOpХ
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp§
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€I *
paddingVALID*
strides
2
Conv2DМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOpИ
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€I 2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
ReluЯ
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€T::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs
г

*__inference_dense_6_layer_call_fn_21511681

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallш
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_215112952
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А	::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:€€€€€€€€€А	
 
_user_specified_nameinputs
Й
Ѕ
'__inference_m_22_layer_call_fn_21511378
conv2d_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCall≤
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_m_22_layer_call_and_return_conditional_losses_215113632
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:€€€€€€€€€ј
)
_user_specified_nameconv2d_12_input
ќ
g
H__inference_dropout_12_layer_call_and_return_conditional_losses_21511588

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeљ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€©*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/y«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€©2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€©2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€©2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€©:X T
0
_output_shapes
:€€€€€€€€€©
 
_user_specified_nameinputs
о
Є
'__inference_m_22_layer_call_fn_21511556

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_m_22_layer_call_and_return_conditional_losses_215114042
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
ґ
O
3__inference_max_pooling2d_13_layer_call_fn_21511150

inputs
identityт
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_215111442
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
ы!
а
B__inference_m_22_layer_call_and_return_conditional_losses_21511336
conv2d_12_input
conv2d_12_21511315
conv2d_12_21511317
conv2d_13_21511322
conv2d_13_21511324
dense_6_21511330
dense_6_21511332
identityИҐ!conv2d_12/StatefulPartitionedCallҐ!conv2d_13/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallі
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputconv2d_12_21511315conv2d_12_21511317*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_215111652#
!conv2d_12/StatefulPartitionedCallМ
dropout_12/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_215111982
dropout_12/PartitionedCallЦ
 max_pooling2d_12/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_215111322"
 max_pooling2d_12/PartitionedCallЌ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_13_21511322conv2d_13_21511324*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_215112232#
!conv2d_13/StatefulPartitionedCallЛ
dropout_13/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_215112562
dropout_13/PartitionedCallЦ
 max_pooling2d_13/PartitionedCallPartitionedCall#dropout_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€$ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_215111442"
 max_pooling2d_13/PartitionedCallА
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_215112762
flatten_6/PartitionedCallі
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_21511330dense_6_21511332*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_215112952!
dense_6/StatefulPartitionedCallж
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:a ]
0
_output_shapes
:€€€€€€€€€ј
)
_user_specified_nameconv2d_12_input
њ
c
G__inference_flatten_6_layer_call_and_return_conditional_losses_21511276

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:€€€€€€€€€А	2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:€€€€€€€€€А	2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€$ :W S
/
_output_shapes
:€€€€€€€€€$ 
 
_user_specified_nameinputs
ЊЛ
ъ
$__inference__traced_restore_21511912
file_prefix%
!assignvariableop_conv2d_12_kernel%
!assignvariableop_1_conv2d_12_bias'
#assignvariableop_2_conv2d_13_kernel%
!assignvariableop_3_conv2d_13_bias%
!assignvariableop_4_dense_6_kernel#
assignvariableop_5_dense_6_bias 
assignvariableop_6_adam_iter"
assignvariableop_7_adam_beta_1"
assignvariableop_8_adam_beta_2!
assignvariableop_9_adam_decay*
&assignvariableop_10_adam_learning_rate
assignvariableop_11_total
assignvariableop_12_count&
"assignvariableop_13_true_positives'
#assignvariableop_14_false_positives(
$assignvariableop_15_true_positives_1'
#assignvariableop_16_false_negatives(
$assignvariableop_17_true_positives_2&
"assignvariableop_18_true_negatives)
%assignvariableop_19_false_positives_1)
%assignvariableop_20_false_negatives_1/
+assignvariableop_21_adam_conv2d_12_kernel_m-
)assignvariableop_22_adam_conv2d_12_bias_m/
+assignvariableop_23_adam_conv2d_13_kernel_m-
)assignvariableop_24_adam_conv2d_13_bias_m-
)assignvariableop_25_adam_dense_6_kernel_m+
'assignvariableop_26_adam_dense_6_bias_m/
+assignvariableop_27_adam_conv2d_12_kernel_v-
)assignvariableop_28_adam_conv2d_12_bias_v/
+assignvariableop_29_adam_conv2d_13_kernel_v-
)assignvariableop_30_adam_conv2d_13_bias_v-
)assignvariableop_31_adam_dense_6_kernel_v+
'assignvariableop_32_adam_dense_6_bias_v
identity_34ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_10ҐAssignVariableOp_11ҐAssignVariableOp_12ҐAssignVariableOp_13ҐAssignVariableOp_14ҐAssignVariableOp_15ҐAssignVariableOp_16ҐAssignVariableOp_17ҐAssignVariableOp_18ҐAssignVariableOp_19ҐAssignVariableOp_2ҐAssignVariableOp_20ҐAssignVariableOp_21ҐAssignVariableOp_22ҐAssignVariableOp_23ҐAssignVariableOp_24ҐAssignVariableOp_25ҐAssignVariableOp_26ҐAssignVariableOp_27ҐAssignVariableOp_28ҐAssignVariableOp_29ҐAssignVariableOp_3ҐAssignVariableOp_30ҐAssignVariableOp_31ҐAssignVariableOp_32ҐAssignVariableOp_4ҐAssignVariableOp_5ҐAssignVariableOp_6ҐAssignVariableOp_7ҐAssignVariableOp_8ҐAssignVariableOp_9Ґ
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ѓ
value§B°"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names“
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesЎ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Ю
_output_shapesЛ
И::::::::::::::::::::::::::::::::::*0
dtypes&
$2"	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity†
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_12_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1¶
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_12_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2®
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_13_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¶
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_13_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4¶
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_6_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5§
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_6_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6°
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7£
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8£
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9Ґ
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10Ѓ
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11°
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12°
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13™
AssignVariableOp_13AssignVariableOp"assignvariableop_13_true_positivesIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14Ђ
AssignVariableOp_14AssignVariableOp#assignvariableop_14_false_positivesIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15ђ
AssignVariableOp_15AssignVariableOp$assignvariableop_15_true_positives_1Identity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16Ђ
AssignVariableOp_16AssignVariableOp#assignvariableop_16_false_negativesIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17ђ
AssignVariableOp_17AssignVariableOp$assignvariableop_17_true_positives_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18™
AssignVariableOp_18AssignVariableOp"assignvariableop_18_true_negativesIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19≠
AssignVariableOp_19AssignVariableOp%assignvariableop_19_false_positives_1Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20≠
AssignVariableOp_20AssignVariableOp%assignvariableop_20_false_negatives_1Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21≥
AssignVariableOp_21AssignVariableOp+assignvariableop_21_adam_conv2d_12_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22±
AssignVariableOp_22AssignVariableOp)assignvariableop_22_adam_conv2d_12_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23≥
AssignVariableOp_23AssignVariableOp+assignvariableop_23_adam_conv2d_13_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24±
AssignVariableOp_24AssignVariableOp)assignvariableop_24_adam_conv2d_13_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25±
AssignVariableOp_25AssignVariableOp)assignvariableop_25_adam_dense_6_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26ѓ
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_6_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27≥
AssignVariableOp_27AssignVariableOp+assignvariableop_27_adam_conv2d_12_kernel_vIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28±
AssignVariableOp_28AssignVariableOp)assignvariableop_28_adam_conv2d_12_bias_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29≥
AssignVariableOp_29AssignVariableOp+assignvariableop_29_adam_conv2d_13_kernel_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30±
AssignVariableOp_30AssignVariableOp)assignvariableop_30_adam_conv2d_13_bias_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31±
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_6_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32ѓ
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_6_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_329
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpі
Identity_33Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_33І
Identity_34IdentityIdentity_33:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_34"#
identity_34Identity_34:output:0*Ы
_input_shapesЙ
Ж: :::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
≠
H
,__inference_flatten_6_layer_call_fn_21511661

inputs
identity…
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_215112762
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:€€€€€€€€€А	2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€$ :W S
/
_output_shapes
:€€€€€€€€€$ 
 
_user_specified_nameinputs
Д
j
N__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_21511144

inputs
identity≠
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€*
ksize
*
paddingVALID*
strides
2	
MaxPoolЗ
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
й
ј
&__inference_signature_wrapper_21511446
conv2d_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCallУ
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *,
f'R%
#__inference__wrapped_model_215111262
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:€€€€€€€€€ј
)
_user_specified_nameconv2d_12_input
К'
њ
B__inference_m_22_layer_call_and_return_conditional_losses_21511522

inputs,
(conv2d_12_conv2d_readvariableop_resource-
)conv2d_12_biasadd_readvariableop_resource,
(conv2d_13_conv2d_readvariableop_resource-
)conv2d_13_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource
identityИҐ conv2d_12/BiasAdd/ReadVariableOpҐconv2d_12/Conv2D/ReadVariableOpҐ conv2d_13/BiasAdd/ReadVariableOpҐconv2d_13/Conv2D/ReadVariableOpҐdense_6/BiasAdd/ReadVariableOpҐdense_6/MatMul/ReadVariableOp≥
conv2d_12/Conv2D/ReadVariableOpReadVariableOp(conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02!
conv2d_12/Conv2D/ReadVariableOp√
conv2d_12/Conv2DConv2Dinputs'conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€©*
paddingVALID*
strides
2
conv2d_12/Conv2D™
 conv2d_12/BiasAdd/ReadVariableOpReadVariableOp)conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_12/BiasAdd/ReadVariableOp±
conv2d_12/BiasAddBiasAddconv2d_12/Conv2D:output:0(conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€©2
conv2d_12/BiasAdd
conv2d_12/ReluReluconv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2
conv2d_12/ReluП
dropout_12/IdentityIdentityconv2d_12/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€©2
dropout_12/Identity 
max_pooling2d_12/MaxPoolMaxPooldropout_12/Identity:output:0*/
_output_shapes
:€€€€€€€€€T*
ksize
*
paddingVALID*
strides
2
max_pooling2d_12/MaxPool≥
conv2d_13/Conv2D/ReadVariableOpReadVariableOp(conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_13/Conv2D/ReadVariableOpЁ
conv2d_13/Conv2DConv2D!max_pooling2d_12/MaxPool:output:0'conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€I *
paddingVALID*
strides
2
conv2d_13/Conv2D™
 conv2d_13/BiasAdd/ReadVariableOpReadVariableOp)conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_13/BiasAdd/ReadVariableOp∞
conv2d_13/BiasAddBiasAddconv2d_13/Conv2D:output:0(conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
conv2d_13/BiasAdd~
conv2d_13/ReluReluconv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
conv2d_13/ReluО
dropout_13/IdentityIdentityconv2d_13/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
dropout_13/Identity 
max_pooling2d_13/MaxPoolMaxPooldropout_13/Identity:output:0*/
_output_shapes
:€€€€€€€€€$ *
ksize
*
paddingVALID*
strides
2
max_pooling2d_13/MaxPools
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
flatten_6/Const°
flatten_6/ReshapeReshape!max_pooling2d_13/MaxPool:output:0flatten_6/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А	2
flatten_6/Reshape¶
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource*
_output_shapes
:	А	X*
dtype02
dense_6/MatMul/ReadVariableOpЯ
dense_6/MatMulMatMulflatten_6/Reshape:output:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€X2
dense_6/MatMul§
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02 
dense_6/BiasAdd/ReadVariableOp°
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€X2
dense_6/BiasAddy
dense_6/SigmoidSigmoiddense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€X2
dense_6/Sigmoid≤
IdentityIdentitydense_6/Sigmoid:y:0!^conv2d_12/BiasAdd/ReadVariableOp ^conv2d_12/Conv2D/ReadVariableOp!^conv2d_13/BiasAdd/ReadVariableOp ^conv2d_13/Conv2D/ReadVariableOp^dense_6/BiasAdd/ReadVariableOp^dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::2D
 conv2d_12/BiasAdd/ReadVariableOp conv2d_12/BiasAdd/ReadVariableOp2B
conv2d_12/Conv2D/ReadVariableOpconv2d_12/Conv2D/ReadVariableOp2D
 conv2d_13/BiasAdd/ReadVariableOp conv2d_13/BiasAdd/ReadVariableOp2B
conv2d_13/Conv2D/ReadVariableOpconv2d_13/Conv2D/ReadVariableOp2@
dense_6/BiasAdd/ReadVariableOpdense_6/BiasAdd/ReadVariableOp2>
dense_6/MatMul/ReadVariableOpdense_6/MatMul/ReadVariableOp:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
л
f
H__inference_dropout_13_layer_call_and_return_conditional_losses_21511640

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identity_1"!

identity_1Identity_1:output:0*.
_input_shapes
:€€€€€€€€€I :W S
/
_output_shapes
:€€€€€€€€€I 
 
_user_specified_nameinputs
ќ
g
H__inference_dropout_12_layer_call_and_return_conditional_losses_21511193

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Const|
dropout/MulMulinputsdropout/Const:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shapeљ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*0
_output_shapes
:€€€€€€€€€©*
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/y«
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2
dropout/GreaterEqualИ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*0
_output_shapes
:€€€€€€€€€©2
dropout/CastГ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*0
_output_shapes
:€€€€€€€€€©2
dropout/Mul_1n
IdentityIdentitydropout/Mul_1:z:0*
T0*0
_output_shapes
:€€€€€€€€€©2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€©:X T
0
_output_shapes
:€€€€€€€€€©
 
_user_specified_nameinputs
ф	
ё
E__inference_dense_6_layer_call_and_return_conditional_losses_21511295

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identityИҐBiasAdd/ReadVariableOpҐMatMul/ReadVariableOpО
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	А	X*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€X2
MatMulМ
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:X*
dtype02
BiasAdd/ReadVariableOpБ
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€X2	
BiasAdda
SigmoidSigmoidBiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€X2	
SigmoidР
IdentityIdentitySigmoid:y:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€А	::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:€€€€€€€€€А	
 
_user_specified_nameinputs
∆
g
H__inference_dropout_13_layer_call_and_return_conditional_losses_21511635

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЉ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€I *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/y∆
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€I 2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€I :W S
/
_output_shapes
:€€€€€€€€€I 
 
_user_specified_nameinputs
ОH
∆
!__inference__traced_save_21511803
file_prefix/
+savev2_conv2d_12_kernel_read_readvariableop-
)savev2_conv2d_12_bias_read_readvariableop/
+savev2_conv2d_13_kernel_read_readvariableop-
)savev2_conv2d_13_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop-
)savev2_true_positives_read_readvariableop.
*savev2_false_positives_read_readvariableop/
+savev2_true_positives_1_read_readvariableop.
*savev2_false_negatives_read_readvariableop/
+savev2_true_positives_2_read_readvariableop-
)savev2_true_negatives_read_readvariableop0
,savev2_false_positives_1_read_readvariableop0
,savev2_false_negatives_1_read_readvariableop6
2savev2_adam_conv2d_12_kernel_m_read_readvariableop4
0savev2_adam_conv2d_12_bias_m_read_readvariableop6
2savev2_adam_conv2d_13_kernel_m_read_readvariableop4
0savev2_adam_conv2d_13_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop6
2savev2_adam_conv2d_12_kernel_v_read_readvariableop4
0savev2_adam_conv2d_12_bias_v_read_readvariableop6
2savev2_adam_conv2d_13_kernel_v_read_readvariableop4
0savev2_adam_conv2d_13_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop
savev2_const

identity_1ИҐMergeV2CheckpointsП
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1Л
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilenameЬ
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*Ѓ
value§B°"B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/1/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/1/false_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/2/true_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/2/false_negatives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_positives/.ATTRIBUTES/VARIABLE_VALUEB=keras_api/metrics/3/true_negatives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_positives/.ATTRIBUTES/VARIABLE_VALUEB>keras_api/metrics/3/false_negatives/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_namesћ
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:"*
dtype0*W
valueNBL"B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slicesѓ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_12_kernel_read_readvariableop)savev2_conv2d_12_bias_read_readvariableop+savev2_conv2d_13_kernel_read_readvariableop)savev2_conv2d_13_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop)savev2_true_positives_read_readvariableop*savev2_false_positives_read_readvariableop+savev2_true_positives_1_read_readvariableop*savev2_false_negatives_read_readvariableop+savev2_true_positives_2_read_readvariableop)savev2_true_negatives_read_readvariableop,savev2_false_positives_1_read_readvariableop,savev2_false_negatives_1_read_readvariableop2savev2_adam_conv2d_12_kernel_m_read_readvariableop0savev2_adam_conv2d_12_bias_m_read_readvariableop2savev2_adam_conv2d_13_kernel_m_read_readvariableop0savev2_adam_conv2d_13_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop2savev2_adam_conv2d_12_kernel_v_read_readvariableop0savev2_adam_conv2d_12_bias_v_read_readvariableop2savev2_adam_conv2d_13_kernel_v_read_readvariableop0savev2_adam_conv2d_13_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *0
dtypes&
$2"	2
SaveV2Ї
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes°
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*Ю
_input_shapesМ
Й: ::: : :	А	X:X: : : : : : : :::::»:»:»:»::: : :	А	X:X::: : :	А	X:X: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	А	X: 

_output_shapes
:X:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
:: 

_output_shapes
::!

_output_shapes	
:»:!

_output_shapes	
:»:!

_output_shapes	
:»:!

_output_shapes	
:»:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :%!

_output_shapes
:	А	X: 

_output_shapes
:X:,(
&
_output_shapes
:: 

_output_shapes
::,(
&
_output_shapes
: : 

_output_shapes
: :% !

_output_shapes
:	А	X: !

_output_shapes
:X:"

_output_shapes
: 
ш*
е
#__inference__wrapped_model_21511126
conv2d_12_input1
-m_22_conv2d_12_conv2d_readvariableop_resource2
.m_22_conv2d_12_biasadd_readvariableop_resource1
-m_22_conv2d_13_conv2d_readvariableop_resource2
.m_22_conv2d_13_biasadd_readvariableop_resource/
+m_22_dense_6_matmul_readvariableop_resource0
,m_22_dense_6_biasadd_readvariableop_resource
identityИҐ%m_22/conv2d_12/BiasAdd/ReadVariableOpҐ$m_22/conv2d_12/Conv2D/ReadVariableOpҐ%m_22/conv2d_13/BiasAdd/ReadVariableOpҐ$m_22/conv2d_13/Conv2D/ReadVariableOpҐ#m_22/dense_6/BiasAdd/ReadVariableOpҐ"m_22/dense_6/MatMul/ReadVariableOp¬
$m_22/conv2d_12/Conv2D/ReadVariableOpReadVariableOp-m_22_conv2d_12_conv2d_readvariableop_resource*&
_output_shapes
:*
dtype02&
$m_22/conv2d_12/Conv2D/ReadVariableOpџ
m_22/conv2d_12/Conv2DConv2Dconv2d_12_input,m_22/conv2d_12/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€©*
paddingVALID*
strides
2
m_22/conv2d_12/Conv2Dє
%m_22/conv2d_12/BiasAdd/ReadVariableOpReadVariableOp.m_22_conv2d_12_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%m_22/conv2d_12/BiasAdd/ReadVariableOp≈
m_22/conv2d_12/BiasAddBiasAddm_22/conv2d_12/Conv2D:output:0-m_22/conv2d_12/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:€€€€€€€€€©2
m_22/conv2d_12/BiasAddО
m_22/conv2d_12/ReluRelum_22/conv2d_12/BiasAdd:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2
m_22/conv2d_12/ReluЮ
m_22/dropout_12/IdentityIdentity!m_22/conv2d_12/Relu:activations:0*
T0*0
_output_shapes
:€€€€€€€€€©2
m_22/dropout_12/Identityў
m_22/max_pooling2d_12/MaxPoolMaxPool!m_22/dropout_12/Identity:output:0*/
_output_shapes
:€€€€€€€€€T*
ksize
*
paddingVALID*
strides
2
m_22/max_pooling2d_12/MaxPool¬
$m_22/conv2d_13/Conv2D/ReadVariableOpReadVariableOp-m_22_conv2d_13_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$m_22/conv2d_13/Conv2D/ReadVariableOpс
m_22/conv2d_13/Conv2DConv2D&m_22/max_pooling2d_12/MaxPool:output:0,m_22/conv2d_13/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€I *
paddingVALID*
strides
2
m_22/conv2d_13/Conv2Dє
%m_22/conv2d_13/BiasAdd/ReadVariableOpReadVariableOp.m_22_conv2d_13_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%m_22/conv2d_13/BiasAdd/ReadVariableOpƒ
m_22/conv2d_13/BiasAddBiasAddm_22/conv2d_13/Conv2D:output:0-m_22/conv2d_13/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
m_22/conv2d_13/BiasAddН
m_22/conv2d_13/ReluRelum_22/conv2d_13/BiasAdd:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
m_22/conv2d_13/ReluЭ
m_22/dropout_13/IdentityIdentity!m_22/conv2d_13/Relu:activations:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
m_22/dropout_13/Identityў
m_22/max_pooling2d_13/MaxPoolMaxPool!m_22/dropout_13/Identity:output:0*/
_output_shapes
:€€€€€€€€€$ *
ksize
*
paddingVALID*
strides
2
m_22/max_pooling2d_13/MaxPool}
m_22/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€А  2
m_22/flatten_6/Constµ
m_22/flatten_6/ReshapeReshape&m_22/max_pooling2d_13/MaxPool:output:0m_22/flatten_6/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€А	2
m_22/flatten_6/Reshapeµ
"m_22/dense_6/MatMul/ReadVariableOpReadVariableOp+m_22_dense_6_matmul_readvariableop_resource*
_output_shapes
:	А	X*
dtype02$
"m_22/dense_6/MatMul/ReadVariableOp≥
m_22/dense_6/MatMulMatMulm_22/flatten_6/Reshape:output:0*m_22/dense_6/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€X2
m_22/dense_6/MatMul≥
#m_22/dense_6/BiasAdd/ReadVariableOpReadVariableOp,m_22_dense_6_biasadd_readvariableop_resource*
_output_shapes
:X*
dtype02%
#m_22/dense_6/BiasAdd/ReadVariableOpµ
m_22/dense_6/BiasAddBiasAddm_22/dense_6/MatMul:product:0+m_22/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€X2
m_22/dense_6/BiasAddИ
m_22/dense_6/SigmoidSigmoidm_22/dense_6/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€X2
m_22/dense_6/Sigmoid’
IdentityIdentitym_22/dense_6/Sigmoid:y:0&^m_22/conv2d_12/BiasAdd/ReadVariableOp%^m_22/conv2d_12/Conv2D/ReadVariableOp&^m_22/conv2d_13/BiasAdd/ReadVariableOp%^m_22/conv2d_13/Conv2D/ReadVariableOp$^m_22/dense_6/BiasAdd/ReadVariableOp#^m_22/dense_6/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::2N
%m_22/conv2d_12/BiasAdd/ReadVariableOp%m_22/conv2d_12/BiasAdd/ReadVariableOp2L
$m_22/conv2d_12/Conv2D/ReadVariableOp$m_22/conv2d_12/Conv2D/ReadVariableOp2N
%m_22/conv2d_13/BiasAdd/ReadVariableOp%m_22/conv2d_13/BiasAdd/ReadVariableOp2L
$m_22/conv2d_13/Conv2D/ReadVariableOp$m_22/conv2d_13/Conv2D/ReadVariableOp2J
#m_22/dense_6/BiasAdd/ReadVariableOp#m_22/dense_6/BiasAdd/ReadVariableOp2H
"m_22/dense_6/MatMul/ReadVariableOp"m_22/dense_6/MatMul/ReadVariableOp:a ]
0
_output_shapes
:€€€€€€€€€ј
)
_user_specified_nameconv2d_12_input
∆
g
H__inference_dropout_13_layer_call_and_return_conditional_losses_21511251

inputs
identityИc
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *д8О?2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/ShapeЉ
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:€€€€€€€€€I *
dtype02&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *Ќћћ=2
dropout/GreaterEqual/y∆
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
dropout/GreaterEqualЗ
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:€€€€€€€€€I 2
dropout/CastВ
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:€€€€€€€€€I 2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identity"
identityIdentity:output:0*.
_input_shapes
:€€€€€€€€€I :W S
/
_output_shapes
:€€€€€€€€€I 
 
_user_specified_nameinputs
И%
™
B__inference_m_22_layer_call_and_return_conditional_losses_21511312
conv2d_12_input
conv2d_12_21511176
conv2d_12_21511178
conv2d_13_21511234
conv2d_13_21511236
dense_6_21511306
dense_6_21511308
identityИҐ!conv2d_12/StatefulPartitionedCallҐ!conv2d_13/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐ"dropout_12/StatefulPartitionedCallҐ"dropout_13/StatefulPartitionedCallі
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputconv2d_12_21511176conv2d_12_21511178*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_215111652#
!conv2d_12/StatefulPartitionedCall§
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_215111932$
"dropout_12/StatefulPartitionedCallЮ
 max_pooling2d_12/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_215111322"
 max_pooling2d_12/PartitionedCallЌ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_13_21511234conv2d_13_21511236*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_215112232#
!conv2d_13/StatefulPartitionedCall»
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_215112512$
"dropout_13/StatefulPartitionedCallЮ
 max_pooling2d_13/PartitionedCallPartitionedCall+dropout_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€$ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_215111442"
 max_pooling2d_13/PartitionedCallА
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_215112762
flatten_6/PartitionedCallі
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_21511306dense_6_21511308*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_215112952!
dense_6/StatefulPartitionedCall∞
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall:a ]
0
_output_shapes
:€€€€€€€€€ј
)
_user_specified_nameconv2d_12_input
н$
°
B__inference_m_22_layer_call_and_return_conditional_losses_21511363

inputs
conv2d_12_21511342
conv2d_12_21511344
conv2d_13_21511349
conv2d_13_21511351
dense_6_21511357
dense_6_21511359
identityИҐ!conv2d_12/StatefulPartitionedCallҐ!conv2d_13/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallҐ"dropout_12/StatefulPartitionedCallҐ"dropout_13/StatefulPartitionedCallЂ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_21511342conv2d_12_21511344*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_215111652#
!conv2d_12/StatefulPartitionedCall§
"dropout_12/StatefulPartitionedCallStatefulPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_215111932$
"dropout_12/StatefulPartitionedCallЮ
 max_pooling2d_12/PartitionedCallPartitionedCall+dropout_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_215111322"
 max_pooling2d_12/PartitionedCallЌ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_13_21511349conv2d_13_21511351*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_215112232#
!conv2d_13/StatefulPartitionedCall»
"dropout_13/StatefulPartitionedCallStatefulPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0#^dropout_12/StatefulPartitionedCall*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_215112512$
"dropout_13/StatefulPartitionedCallЮ
 max_pooling2d_13/PartitionedCallPartitionedCall+dropout_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€$ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_215111442"
 max_pooling2d_13/PartitionedCallА
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_215112762
flatten_6/PartitionedCallі
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_21511357dense_6_21511359*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_215112952!
dense_6/StatefulPartitionedCall∞
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall#^dropout_12/StatefulPartitionedCall#^dropout_13/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2H
"dropout_12/StatefulPartitionedCall"dropout_12/StatefulPartitionedCall2H
"dropout_13/StatefulPartitionedCall"dropout_13/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
о
Є
'__inference_m_22_layer_call_fn_21511539

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCall©
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_m_22_layer_call_and_return_conditional_losses_215113632
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
а!
„
B__inference_m_22_layer_call_and_return_conditional_losses_21511404

inputs
conv2d_12_21511383
conv2d_12_21511385
conv2d_13_21511390
conv2d_13_21511392
dense_6_21511398
dense_6_21511400
identityИҐ!conv2d_12/StatefulPartitionedCallҐ!conv2d_13/StatefulPartitionedCallҐdense_6/StatefulPartitionedCallЂ
!conv2d_12/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_12_21511383conv2d_12_21511385*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_12_layer_call_and_return_conditional_losses_215111652#
!conv2d_12/StatefulPartitionedCallМ
dropout_12/PartitionedCallPartitionedCall*conv2d_12/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_215111982
dropout_12/PartitionedCallЦ
 max_pooling2d_12/PartitionedCallPartitionedCall#dropout_12/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€T* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_215111322"
 max_pooling2d_12/PartitionedCallЌ
!conv2d_13/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_12/PartitionedCall:output:0conv2d_13_21511390conv2d_13_21511392*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_215112232#
!conv2d_13/StatefulPartitionedCallЛ
dropout_13/PartitionedCallPartitionedCall*conv2d_13/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_13_layer_call_and_return_conditional_losses_215112562
dropout_13/PartitionedCallЦ
 max_pooling2d_13/PartitionedCallPartitionedCall#dropout_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€$ * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_215111442"
 max_pooling2d_13/PartitionedCallА
flatten_6/PartitionedCallPartitionedCall)max_pooling2d_13/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:€€€€€€€€€А	* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_flatten_6_layer_call_and_return_conditional_losses_215112762
flatten_6/PartitionedCallі
dense_6/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_6_21511398dense_6_21511400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_dense_6_layer_call_and_return_conditional_losses_215112952!
dense_6/StatefulPartitionedCallж
IdentityIdentity(dense_6/StatefulPartitionedCall:output:0"^conv2d_12/StatefulPartitionedCall"^conv2d_13/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::2F
!conv2d_12/StatefulPartitionedCall!conv2d_12/StatefulPartitionedCall2F
!conv2d_13/StatefulPartitionedCall!conv2d_13/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall:X T
0
_output_shapes
:€€€€€€€€€ј
 
_user_specified_nameinputs
Ѕ
I
-__inference_dropout_12_layer_call_fn_21511603

inputs
identity“
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:€€€€€€€€€©* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_dropout_12_layer_call_and_return_conditional_losses_215111982
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:€€€€€€€€€©2

Identity"
identityIdentity:output:0*/
_input_shapes
:€€€€€€€€€©:X T
0
_output_shapes
:€€€€€€€€€©
 
_user_specified_nameinputs
ґ
O
3__inference_max_pooling2d_12_layer_call_fn_21511138

inputs
identityт
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *W
fRRP
N__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_215111322
PartitionedCallП
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€:r n
J
_output_shapes8
6:4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
 
_user_specified_nameinputs
Й
Ѕ
'__inference_m_22_layer_call_fn_21511419
conv2d_12_input
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
identityИҐStatefulPartitionedCall≤
StatefulPartitionedCallStatefulPartitionedCallconv2d_12_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:€€€€€€€€€X*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_m_22_layer_call_and_return_conditional_losses_215114042
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€X2

Identity"
identityIdentity:output:0*G
_input_shapes6
4:€€€€€€€€€ј::::::22
StatefulPartitionedCallStatefulPartitionedCall:a ]
0
_output_shapes
:€€€€€€€€€ј
)
_user_specified_nameconv2d_12_input
Ж
Б
,__inference_conv2d_13_layer_call_fn_21511623

inputs
unknown
	unknown_0
identityИҐStatefulPartitionedCallВ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:€€€€€€€€€I *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_conv2d_13_layer_call_and_return_conditional_losses_215112232
StatefulPartitionedCallЦ
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:€€€€€€€€€I 2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:€€€€€€€€€T::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:€€€€€€€€€T
 
_user_specified_nameinputs"±L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*√
serving_defaultѓ
T
conv2d_12_inputA
!serving_default_conv2d_12_input:0€€€€€€€€€ј;
dense_60
StatefulPartitionedCall:0€€€€€€€€€Xtensorflow/serving/predict:л…
Ю^
layer_with_weights-0
layer-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer-5
layer-6
layer_with_weights-2
layer-7
		optimizer

regularization_losses
trainable_variables
	variables
	keras_api

signatures
Л_default_save_signature
+М&call_and_return_all_conditional_losses
Н__call__"Ъ[
_tf_keras_sequentialыZ{"class_name": "Sequential", "name": "m_22", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "m_22", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_12_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 5, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "m_22", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "conv2d_12_input"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Conv2D", "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dropout", "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}}, {"class_name": "Flatten", "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}, "training_config": {"loss": {"class_name": "BinaryCrossentropy", "config": {"reduction": "auto", "name": "binary_crossentropy", "from_logits": false, "label_smoothing": 0}}, "metrics": [[{"class_name": "Precision", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "Recall", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}, {"class_name": "AUC", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0006000000284984708, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
ч


kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
+О&call_and_return_all_conditional_losses
П__call__"–	
_tf_keras_layerґ	{"class_name": "Conv2D", "name": "conv2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_12", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 192, 5, 1]}, "dtype": "float32", "filters": 16, "kernel_size": {"class_name": "__tuple__", "items": [24, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 192, 5, 1]}}
й
regularization_losses
trainable_variables
	variables
	keras_api
+Р&call_and_return_all_conditional_losses
С__call__"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_12", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_12", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
Г
regularization_losses
trainable_variables
	variables
	keras_api
+Т&call_and_return_all_conditional_losses
У__call__"т
_tf_keras_layerЎ{"class_name": "MaxPooling2D", "name": "max_pooling2d_12", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_12", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
ч	

kernel
bias
regularization_losses
 trainable_variables
!	variables
"	keras_api
+Ф&call_and_return_all_conditional_losses
Х__call__"–
_tf_keras_layerґ{"class_name": "Conv2D", "name": "conv2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_13", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [12, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 16}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 84, 3, 16]}}
й
#regularization_losses
$trainable_variables
%	variables
&	keras_api
+Ц&call_and_return_all_conditional_losses
Ч__call__"Ў
_tf_keras_layerЊ{"class_name": "Dropout", "name": "dropout_13", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dropout_13", "trainable": true, "dtype": "float32", "rate": 0.1, "noise_shape": null, "seed": null}}
Г
'regularization_losses
(trainable_variables
)	variables
*	keras_api
+Ш&call_and_return_all_conditional_losses
Щ__call__"т
_tf_keras_layerЎ{"class_name": "MaxPooling2D", "name": "max_pooling2d_13", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "max_pooling2d_13", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 1]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 1]}, "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}}
и
+regularization_losses
,trainable_variables
-	variables
.	keras_api
+Ъ&call_and_return_all_conditional_losses
Ы__call__"„
_tf_keras_layerљ{"class_name": "Flatten", "name": "flatten_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten_6", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
щ

/kernel
0bias
1regularization_losses
2trainable_variables
3	variables
4	keras_api
+Ь&call_and_return_all_conditional_losses
Э__call__"“
_tf_keras_layerЄ{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 88, "activation": "sigmoid", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1152}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1152]}}
 
5iter

6beta_1

7beta_2
	8decay
9learning_ratemmАmБmВ/mГ0mДvЕvЖvЗvИ/vЙ0vК"
	optimizer
 "
trackable_list_wrapper
J
0
1
2
3
/4
05"
trackable_list_wrapper
J
0
1
2
3
/4
05"
trackable_list_wrapper
ќ

regularization_losses
:non_trainable_variables
;layer_regularization_losses
<metrics

=layers
>layer_metrics
trainable_variables
	variables
Н__call__
Л_default_save_signature
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
-
Юserving_default"
signature_map
*:(2conv2d_12/kernel
:2conv2d_12/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
regularization_losses
?non_trainable_variables
@layer_regularization_losses
Ametrics

Blayers
Clayer_metrics
trainable_variables
	variables
П__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
regularization_losses
Dnon_trainable_variables
Elayer_regularization_losses
Fmetrics

Glayers
Hlayer_metrics
trainable_variables
	variables
С__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
regularization_losses
Inon_trainable_variables
Jlayer_regularization_losses
Kmetrics

Llayers
Mlayer_metrics
trainable_variables
	variables
У__call__
+Т&call_and_return_all_conditional_losses
'Т"call_and_return_conditional_losses"
_generic_user_object
*:( 2conv2d_13/kernel
: 2conv2d_13/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
∞
regularization_losses
Nnon_trainable_variables
Olayer_regularization_losses
Pmetrics

Qlayers
Rlayer_metrics
 trainable_variables
!	variables
Х__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
#regularization_losses
Snon_trainable_variables
Tlayer_regularization_losses
Umetrics

Vlayers
Wlayer_metrics
$trainable_variables
%	variables
Ч__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
'regularization_losses
Xnon_trainable_variables
Ylayer_regularization_losses
Zmetrics

[layers
\layer_metrics
(trainable_variables
)	variables
Щ__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
∞
+regularization_losses
]non_trainable_variables
^layer_regularization_losses
_metrics

`layers
alayer_metrics
,trainable_variables
-	variables
Ы__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
!:	А	X2dense_6/kernel
:X2dense_6/bias
 "
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
∞
1regularization_losses
bnon_trainable_variables
clayer_regularization_losses
dmetrics

elayers
flayer_metrics
2trainable_variables
3	variables
Э__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
g0
h1
i2
j3"
trackable_list_wrapper
X
0
1
2
3
4
5
6
7"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ї
	ktotal
	lcount
m	variables
n	keras_api"Д
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
£
o
thresholds
ptrue_positives
qfalse_positives
r	variables
s	keras_api"…
_tf_keras_metricЃ{"class_name": "Precision", "name": "precision", "dtype": "float32", "config": {"name": "precision", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
Ъ
t
thresholds
utrue_positives
vfalse_negatives
w	variables
x	keras_api"ј
_tf_keras_metric•{"class_name": "Recall", "name": "recall", "dtype": "float32", "config": {"name": "recall", "dtype": "float32", "thresholds": null, "top_k": null, "class_id": null}}
ѓ"
ytrue_positives
ztrue_negatives
{false_positives
|false_negatives
}	variables
~	keras_api"Љ!
_tf_keras_metric°!{"class_name": "AUC", "name": "auc", "dtype": "float32", "config": {"name": "auc", "dtype": "float32", "num_thresholds": 200, "curve": "ROC", "summation_method": "interpolation", "thresholds": [0.005025125628140704, 0.010050251256281407, 0.01507537688442211, 0.020100502512562814, 0.02512562814070352, 0.03015075376884422, 0.035175879396984924, 0.04020100502512563, 0.04522613065326633, 0.05025125628140704, 0.05527638190954774, 0.06030150753768844, 0.06532663316582915, 0.07035175879396985, 0.07537688442211055, 0.08040201005025126, 0.08542713567839195, 0.09045226130653267, 0.09547738693467336, 0.10050251256281408, 0.10552763819095477, 0.11055276381909548, 0.11557788944723618, 0.12060301507537688, 0.12562814070351758, 0.1306532663316583, 0.135678391959799, 0.1407035175879397, 0.1457286432160804, 0.1507537688442211, 0.15577889447236182, 0.16080402010050251, 0.1658291457286432, 0.1708542713567839, 0.17587939698492464, 0.18090452261306533, 0.18592964824120603, 0.19095477386934673, 0.19597989949748743, 0.20100502512562815, 0.20603015075376885, 0.21105527638190955, 0.21608040201005024, 0.22110552763819097, 0.22613065326633167, 0.23115577889447236, 0.23618090452261306, 0.24120603015075376, 0.24623115577889448, 0.25125628140703515, 0.2562814070351759, 0.2613065326633166, 0.2663316582914573, 0.271356783919598, 0.27638190954773867, 0.2814070351758794, 0.2864321608040201, 0.2914572864321608, 0.2964824120603015, 0.3015075376884422, 0.3065326633165829, 0.31155778894472363, 0.3165829145728643, 0.32160804020100503, 0.32663316582914576, 0.3316582914572864, 0.33668341708542715, 0.3417085427135678, 0.34673366834170855, 0.35175879396984927, 0.35678391959798994, 0.36180904522613067, 0.36683417085427134, 0.37185929648241206, 0.3768844221105528, 0.38190954773869346, 0.3869346733668342, 0.39195979899497485, 0.3969849246231156, 0.4020100502512563, 0.40703517587939697, 0.4120603015075377, 0.41708542713567837, 0.4221105527638191, 0.4271356783919598, 0.4321608040201005, 0.4371859296482412, 0.44221105527638194, 0.4472361809045226, 0.45226130653266333, 0.457286432160804, 0.4623115577889447, 0.46733668341708545, 0.4723618090452261, 0.47738693467336685, 0.4824120603015075, 0.48743718592964824, 0.49246231155778897, 0.49748743718592964, 0.5025125628140703, 0.507537688442211, 0.5125628140703518, 0.5175879396984925, 0.5226130653266332, 0.5276381909547738, 0.5326633165829145, 0.5376884422110553, 0.542713567839196, 0.5477386934673367, 0.5527638190954773, 0.5577889447236181, 0.5628140703517588, 0.5678391959798995, 0.5728643216080402, 0.5778894472361809, 0.5829145728643216, 0.5879396984924623, 0.592964824120603, 0.5979899497487438, 0.6030150753768844, 0.6080402010050251, 0.6130653266331658, 0.6180904522613065, 0.6231155778894473, 0.628140703517588, 0.6331658291457286, 0.6381909547738693, 0.6432160804020101, 0.6482412060301508, 0.6532663316582915, 0.6582914572864321, 0.6633165829145728, 0.6683417085427136, 0.6733668341708543, 0.678391959798995, 0.6834170854271356, 0.6884422110552764, 0.6934673366834171, 0.6984924623115578, 0.7035175879396985, 0.7085427135678392, 0.7135678391959799, 0.7185929648241206, 0.7236180904522613, 0.7286432160804021, 0.7336683417085427, 0.7386934673366834, 0.7437185929648241, 0.7487437185929648, 0.7537688442211056, 0.7587939698492462, 0.7638190954773869, 0.7688442211055276, 0.7738693467336684, 0.7788944723618091, 0.7839195979899497, 0.7889447236180904, 0.7939698492462312, 0.7989949748743719, 0.8040201005025126, 0.8090452261306532, 0.8140703517587939, 0.8190954773869347, 0.8241206030150754, 0.8291457286432161, 0.8341708542713567, 0.8391959798994975, 0.8442211055276382, 0.8492462311557789, 0.8542713567839196, 0.8592964824120602, 0.864321608040201, 0.8693467336683417, 0.8743718592964824, 0.8793969849246231, 0.8844221105527639, 0.8894472361809045, 0.8944723618090452, 0.8994974874371859, 0.9045226130653267, 0.9095477386934674, 0.914572864321608, 0.9195979899497487, 0.9246231155778895, 0.9296482412060302, 0.9346733668341709, 0.9396984924623115, 0.9447236180904522, 0.949748743718593, 0.9547738693467337, 0.9597989949748744, 0.964824120603015, 0.9698492462311558, 0.9748743718592965, 0.9798994974874372, 0.9849246231155779, 0.9899497487437185, 0.9949748743718593], "multi_label": false, "label_weights": null}}
:  (2total
:  (2count
.
k0
l1"
trackable_list_wrapper
-
m	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_positives
.
p0
q1"
trackable_list_wrapper
-
r	variables"
_generic_user_object
 "
trackable_list_wrapper
: (2true_positives
: (2false_negatives
.
u0
v1"
trackable_list_wrapper
-
w	variables"
_generic_user_object
:» (2true_positives
:» (2true_negatives
 :» (2false_positives
 :» (2false_negatives
<
y0
z1
{2
|3"
trackable_list_wrapper
-
}	variables"
_generic_user_object
/:-2Adam/conv2d_12/kernel/m
!:2Adam/conv2d_12/bias/m
/:- 2Adam/conv2d_13/kernel/m
!: 2Adam/conv2d_13/bias/m
&:$	А	X2Adam/dense_6/kernel/m
:X2Adam/dense_6/bias/m
/:-2Adam/conv2d_12/kernel/v
!:2Adam/conv2d_12/bias/v
/:- 2Adam/conv2d_13/kernel/v
!: 2Adam/conv2d_13/bias/v
&:$	А	X2Adam/dense_6/kernel/v
:X2Adam/dense_6/bias/v
т2п
#__inference__wrapped_model_21511126«
Л≤З
FullArgSpec
argsЪ 
varargsjargs
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *7Ґ4
2К/
conv2d_12_input€€€€€€€€€ј
÷2”
B__inference_m_22_layer_call_and_return_conditional_losses_21511491
B__inference_m_22_layer_call_and_return_conditional_losses_21511312
B__inference_m_22_layer_call_and_return_conditional_losses_21511336
B__inference_m_22_layer_call_and_return_conditional_losses_21511522ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
к2з
'__inference_m_22_layer_call_fn_21511378
'__inference_m_22_layer_call_fn_21511556
'__inference_m_22_layer_call_fn_21511419
'__inference_m_22_layer_call_fn_21511539ј
Ј≤≥
FullArgSpec1
args)Ъ&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЪ
p 

 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
с2о
G__inference_conv2d_12_layer_call_and_return_conditional_losses_21511567Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_conv2d_12_layer_call_fn_21511576Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ2Ћ
H__inference_dropout_12_layer_call_and_return_conditional_losses_21511588
H__inference_dropout_12_layer_call_and_return_conditional_losses_21511593і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ш2Х
-__inference_dropout_12_layer_call_fn_21511598
-__inference_dropout_12_layer_call_fn_21511603і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ґ2≥
N__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_21511132а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ы2Ш
3__inference_max_pooling2d_12_layer_call_fn_21511138а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
с2о
G__inference_conv2d_13_layer_call_and_return_conditional_losses_21511614Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_conv2d_13_layer_call_fn_21511623Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ќ2Ћ
H__inference_dropout_13_layer_call_and_return_conditional_losses_21511635
H__inference_dropout_13_layer_call_and_return_conditional_losses_21511640і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
Ш2Х
-__inference_dropout_13_layer_call_fn_21511645
-__inference_dropout_13_layer_call_fn_21511650і
Ђ≤І
FullArgSpec)
args!Ъ
jself
jinputs

jtraining
varargs
 
varkw
 
defaultsЪ
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ґ2≥
N__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_21511144а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ы2Ш
3__inference_max_pooling2d_13_layer_call_fn_21511150а
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *@Ґ=
;К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
с2о
G__inference_flatten_6_layer_call_and_return_conditional_losses_21511656Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
÷2”
,__inference_flatten_6_layer_call_fn_21511661Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
п2м
E__inference_dense_6_layer_call_and_return_conditional_losses_21511672Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
‘2—
*__inference_dense_6_layer_call_fn_21511681Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
’B“
&__inference_signature_wrapper_21511446conv2d_12_input"Ф
Н≤Й
FullArgSpec
argsЪ 
varargs
 
varkwjkwargs
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 •
#__inference__wrapped_model_21511126~/0AҐ>
7Ґ4
2К/
conv2d_12_input€€€€€€€€€ј
™ "1™.
,
dense_6!К
dense_6€€€€€€€€€Xє
G__inference_conv2d_12_layer_call_and_return_conditional_losses_21511567n8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ј
™ ".Ґ+
$К!
0€€€€€€€€€©
Ъ С
,__inference_conv2d_12_layer_call_fn_21511576a8Ґ5
.Ґ+
)К&
inputs€€€€€€€€€ј
™ "!К€€€€€€€€€©Ј
G__inference_conv2d_13_layer_call_and_return_conditional_losses_21511614l7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€T
™ "-Ґ*
#К 
0€€€€€€€€€I 
Ъ П
,__inference_conv2d_13_layer_call_fn_21511623_7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€T
™ " К€€€€€€€€€I ¶
E__inference_dense_6_layer_call_and_return_conditional_losses_21511672]/00Ґ-
&Ґ#
!К
inputs€€€€€€€€€А	
™ "%Ґ"
К
0€€€€€€€€€X
Ъ ~
*__inference_dense_6_layer_call_fn_21511681P/00Ґ-
&Ґ#
!К
inputs€€€€€€€€€А	
™ "К€€€€€€€€€XЇ
H__inference_dropout_12_layer_call_and_return_conditional_losses_21511588n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€©
p
™ ".Ґ+
$К!
0€€€€€€€€€©
Ъ Ї
H__inference_dropout_12_layer_call_and_return_conditional_losses_21511593n<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€©
p 
™ ".Ґ+
$К!
0€€€€€€€€€©
Ъ Т
-__inference_dropout_12_layer_call_fn_21511598a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€©
p
™ "!К€€€€€€€€€©Т
-__inference_dropout_12_layer_call_fn_21511603a<Ґ9
2Ґ/
)К&
inputs€€€€€€€€€©
p 
™ "!К€€€€€€€€€©Є
H__inference_dropout_13_layer_call_and_return_conditional_losses_21511635l;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€I 
p
™ "-Ґ*
#К 
0€€€€€€€€€I 
Ъ Є
H__inference_dropout_13_layer_call_and_return_conditional_losses_21511640l;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€I 
p 
™ "-Ґ*
#К 
0€€€€€€€€€I 
Ъ Р
-__inference_dropout_13_layer_call_fn_21511645_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€I 
p
™ " К€€€€€€€€€I Р
-__inference_dropout_13_layer_call_fn_21511650_;Ґ8
1Ґ.
(К%
inputs€€€€€€€€€I 
p 
™ " К€€€€€€€€€I ђ
G__inference_flatten_6_layer_call_and_return_conditional_losses_21511656a7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€$ 
™ "&Ґ#
К
0€€€€€€€€€А	
Ъ Д
,__inference_flatten_6_layer_call_fn_21511661T7Ґ4
-Ґ*
(К%
inputs€€€€€€€€€$ 
™ "К€€€€€€€€€А	ј
B__inference_m_22_layer_call_and_return_conditional_losses_21511312z/0IҐF
?Ґ<
2К/
conv2d_12_input€€€€€€€€€ј
p

 
™ "%Ґ"
К
0€€€€€€€€€X
Ъ ј
B__inference_m_22_layer_call_and_return_conditional_losses_21511336z/0IҐF
?Ґ<
2К/
conv2d_12_input€€€€€€€€€ј
p 

 
™ "%Ґ"
К
0€€€€€€€€€X
Ъ Ј
B__inference_m_22_layer_call_and_return_conditional_losses_21511491q/0@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ј
p

 
™ "%Ґ"
К
0€€€€€€€€€X
Ъ Ј
B__inference_m_22_layer_call_and_return_conditional_losses_21511522q/0@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ј
p 

 
™ "%Ґ"
К
0€€€€€€€€€X
Ъ Ш
'__inference_m_22_layer_call_fn_21511378m/0IҐF
?Ґ<
2К/
conv2d_12_input€€€€€€€€€ј
p

 
™ "К€€€€€€€€€XШ
'__inference_m_22_layer_call_fn_21511419m/0IҐF
?Ґ<
2К/
conv2d_12_input€€€€€€€€€ј
p 

 
™ "К€€€€€€€€€XП
'__inference_m_22_layer_call_fn_21511539d/0@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ј
p

 
™ "К€€€€€€€€€XП
'__inference_m_22_layer_call_fn_21511556d/0@Ґ=
6Ґ3
)К&
inputs€€€€€€€€€ј
p 

 
™ "К€€€€€€€€€Xс
N__inference_max_pooling2d_12_layer_call_and_return_conditional_losses_21511132ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ …
3__inference_max_pooling2d_12_layer_call_fn_21511138СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€с
N__inference_max_pooling2d_13_layer_call_and_return_conditional_losses_21511144ЮRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ "HҐE
>К;
04€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
Ъ …
3__inference_max_pooling2d_13_layer_call_fn_21511150СRҐO
HҐE
CК@
inputs4€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€
™ ";К84€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€€Љ
&__inference_signature_wrapper_21511446С/0TҐQ
Ґ 
J™G
E
conv2d_12_input2К/
conv2d_12_input€€€€€€€€€ј"1™.
,
dense_6!К
dense_6€€€€€€€€€X