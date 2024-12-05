const mongoose = require('mongoose')
const db = require('./db.js')
const bcrypt = require('bcryptjs');

const memberSchema = mongoose.Schema({
  name: String,
  school: String,
  grade: Number,
  science: Number,
  tech: Number,
  math: Number,
  codingClass: { type: String, enum: ['Ya', 'Tidak'], default: 'Ya' }
})

const nonCodingStudentSchema = new mongoose.Schema({
  name: String,
  school: String,
  grade: Number,
  science: Number,
  tech: Number,
  math: Number,
  codingClass: { type: String, enum: ['Ya', 'Tidak'], default: 'Tidak' }
});

const userSchema = new mongoose.Schema({
  name: String,
  email: String,
  password: String,
});

userSchema.pre('save', async function (next) {
  if (!this.isModified('password')) return next();
  const salt = await bcrypt.genSalt(10);
  this.password = await bcrypt.hash(this.password, salt);
  next();
});
const Member = mongoose.model('Member', memberSchema)
const NonCodingStudent = mongoose.model('NonCodingStudent', nonCodingStudentSchema);
const User = mongoose.model('User', userSchema)
module.exports = { Member, NonCodingStudent, User };


// async function createUser() {
//   try{
//     const newUser = await User.create({
//     name: 'Greg',
//     email: 'Greg@gmail.com',
//     password: 'password1',
//   });
//     console.log('Data user berhasil disimpan:', newUser)
//   }catch(error) {
//       console.error('Terjadi kesalahan saat menyimpan data user:', error)
//     }
// }
// createUser()