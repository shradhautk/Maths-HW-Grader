import UploadPage from '../../components/UploadPage'

export default function StudentUpload() {
  return (
    <UploadPage
      copy={{
        title: 'Upload Your Assignment',
        subtitle: "Submit your completed work as a PDF and GradeAnt's AI grader takes it from there.",
        dropLabel: 'Drag & drop your file here',
        submitLabel: 'Upload Assignments',
        types: [{ value: 'student', label: 'Student Assignment' }],
        steps: [
          { title: 'Add your IDs', text: 'Enter the Assignment ID and Question Paper ID your teacher shared with you.' },
          { title: 'Drop your PDF', text: 'Scan or export your completed work as a single PDF, then drag it in.' },
          {
            title: 'Get graded fast',
            text: 'GradeAnt reads your answers and posts results to My Grades — usually within minutes.',
          },
        ],
        note: 'Keep handwriting clear and each page right-side up for the most accurate grading.',
      }}
    />
  )
}
