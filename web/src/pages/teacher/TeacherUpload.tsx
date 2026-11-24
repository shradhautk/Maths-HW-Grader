import UploadPage from '../../components/UploadPage'

export default function TeacherUpload() {
  return (
    <UploadPage
      copy={{
        title: 'Upload Question Paper or Solution',
        subtitle:
          "Add the question paper and, if you're grading with a standard solution key, the model answer set.",
        dropLabel: 'Drag & drop the PDF here',
        submitLabel: 'Upload Files',
        types: [
          { value: 'teacher', label: 'Teacher Question Paper' },
          { value: 'answer', label: 'Teacher solution' },
        ],
        steps: [
          {
            title: 'Choose the upload type',
            text: 'Question paper first, then (if using a standard solution) the answer key.',
          },
          {
            title: 'Tag it correctly',
            text: 'Match the Assignment ID and Question Paper ID across both uploads so they link together.',
          },
          { title: 'Let GradeAnt grade', text: 'Once uploaded, review scores anytime from Students Grades.' },
        ],
        note: 'Uploading a solution? Use the same Assignment ID and Question Paper ID as the question paper.',
      }}
    />
  )
}
